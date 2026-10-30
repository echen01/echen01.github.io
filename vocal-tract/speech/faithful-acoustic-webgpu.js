import {createGpuFft} from './faithful-fft-webgpu.js';
import {acousticFftSize} from './faithful-acoustic.js';

// Batched direct port of faithful-acoustic.js. All buffers remain GPU resident.
// The cached primal belongs to the most recent batch; callers doing overlap-add
// over multiple batches must recompute each batch before its backward pass.
// Backward consumes the cached noise spectrum; call forward again before a
// second VJP, even when the inputs have not changed. With compensatedCotangent,
// callers supply separate f32 high/low parts through cotangentBuffer and
// cotangentLowBuffer; their sum is the real cotangent for the paired-input FFT.
// compensatedInput similarly accepts source parts in uinBuffer/uinLowBuffer.
export async function createFaithfulGpuAcoustic(device, options = {}) {
  const sampleRate = options.sampleRate ?? 32000;
  const frames = options.frames ?? 8, n = options.fftSize ?? acousticFftSize(sampleRate), length = options.length ?? 17.5;
  const fftOptions = options.fftOptions ?? {};
  const compensatedAccumulation = options.compensatedAccumulation ?? false;
  const compensatedCotangent = options.compensatedCotangent ?? false;
  const compensatedInput = options.compensatedInput ?? false;
  if ((compensatedCotangent || compensatedInput) && fftOptions.arithmetic !== 'double-single') throw new Error('Compensated acoustic inputs require double-single FFT arithmetic.');
  if (![512, 1024].includes(n) || !Number.isInteger(frames) || frames < 1 || frames > 32) throw new Error('Faithful GPU acoustics requires FFT512 or FFT1024 and a batch of 1–32 frames.');
  const sections = 31, bins = n / 2 + 1, transferCount = frames * bins * (sections * 2 + 1), cacheBlock = frames * bins * sections * 4;
  const buffers = [], usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buffer = (label, scalars) => { const b = device.createBuffer({label, size: scalars * 4, usage}); buffers.push(b); return b; };
  const areaBuffer = buffer('faithful acoustic areas', frames * 32), uinBuffer = buffer('faithful acoustic windowed source', frames * n), noiseBuffer = buffer('faithful acoustic windowed noise', frames * sections * n);
  const outputBuffer = buffer('faithful acoustic frame output', frames * n), cotangentBuffer = buffer('faithful acoustic frame cotangent', frames * n), dlogAreaBuffer = buffer('faithful acoustic log-area gradient', frames * 32);
  const transfer = buffer('faithful acoustic Hc Hfric Htract', transferCount * 2), gtransfer = buffer('faithful acoustic transfer cotangents', transferCount * 2);
  const cache = buffer('faithful acoustic matrices and adjoints', cacheBlock * 6 * 2), lowpassed = buffer('faithful acoustic lowpassed constriction flow', frames * sections * n);
  const gateGrad = buffer('faithful acoustic gate derivatives', frames * sections * 2), frequencyGrad = buffer('faithful acoustic per-frequency area gradients', frames * bins * 32);
  const frameA = buffer('faithful acoustic frame FFT scratch A', frames * n * 2), frameB = buffer('faithful acoustic frame FFT scratch B', frames * n * 2), tubeA = buffer('faithful acoustic tube FFT scratch A', frames * sections * n * 2), tubeB = buffer('faithful acoustic tube FFT scratch B', frames * sections * n * 2), U = buffer('faithful acoustic source spectrum', frames * n * 2);
  const cotangentLowBuffer = compensatedCotangent ? buffer('faithful acoustic cotangent low components', frames * n) : undefined;
  const framePairs = compensatedCotangent || compensatedInput ? buffer('faithful acoustic paired FFT input', frames * n * 4) : undefined;
  const uinLowBuffer = compensatedInput ? buffer('faithful acoustic source low components', frames * n) : undefined;
  const resources = [areaBuffer, uinBuffer, noiseBuffer, outputBuffer, cotangentBuffer, dlogAreaBuffer, transfer, gtransfer, cache, lowpassed, gateGrad, frequencyGrad, frameA, frameB, tubeA, tubeB, U, cotangentLowBuffer, framePairs, uinLowBuffer];
  const shader = /* wgsl */`
const N: u32 = ${n}u; const BINS: u32 = ${bins}u; const FRAMES: u32 = ${frames}u; const S: u32 = 31u;
const BLOCK: u32 = ${cacheBlock}u; const HF: u32 = ${frames * sections * bins}u; const HT: u32 = ${frames * sections * bins * 2}u;
const PI: f32 = 3.141592653589793; const SR: f32 = ${sampleRate}; const LENGTH: f32 = ${length};
const RHO: f32 = 1.14e-3; const MU: f32 = 1.86e-4; const C: f32 = 34300.0;
const RE_SCALE: f32 = 4.0 * RHO * RHO / (PI * MU * MU); const RE_THRESHOLD: f32 = (3500.0 / 400.0) * (3500.0 / 400.0);
const ALPHA: f32 = 2.0 * PI * 2000.0 / SR;
const COMPENSATED: bool = ${compensatedAccumulation ? 'true' : 'false'};
@group(0) @binding(0) var<storage, read> areas: array<f32>;
@group(0) @binding(1) var<storage, read> uin: array<f32>;
@group(0) @binding(2) var<storage, read> noise: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read> cotangent: array<f32>;
@group(0) @binding(5) var<storage, read_write> dlogareas: array<f32>;
@group(0) @binding(6) var<storage, read_write> transfer: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> gtransfer: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> cache: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> lowpassed: array<f32>;
@group(0) @binding(10) var<storage, read_write> gategrad: array<vec2<f32>>;
@group(0) @binding(11) var<storage, read_write> freqgrad: array<f32>;
@group(0) @binding(12) var<storage, read_write> frameA: array<vec2<f32>>;
@group(0) @binding(13) var<storage, read_write> frameB: array<vec2<f32>>;
@group(0) @binding(14) var<storage, read_write> tubeA: array<vec2<f32>>;
@group(0) @binding(15) var<storage, read_write> tubeB: array<vec2<f32>>;
@group(0) @binding(16) var<storage, read_write> U: array<vec2<f32>>;
${compensatedCotangent ? '@group(0) @binding(17) var<storage, read> cotangentLow: array<f32>;' : ''}
${compensatedCotangent || compensatedInput ? '@group(0) @binding(18) var<storage, read_write> framePairs: array<vec4<f32>>;' : ''}
${compensatedInput ? '@group(0) @binding(19) var<storage, read> uinLow: array<f32>;' : ''}
fn conj(a: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x, -a.y); }
fn mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
fn div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return mul(a,conj(b))/dot(b,b); }
// Neumaier summation preserves low-order bits when signed acoustic terms
// cancel. The opt-in path changes evaluation accuracy, never the objective.
fn accumulate(sum:ptr<function,f32>,error:ptr<function,f32>,term:f32){
  if(COMPENSATED){let next=(*sum)+term;(*error)+=select((term-next)+(*sum),((*sum)-next)+term,abs(*sum)>=abs(term));(*sum)=next;}
  else{(*sum)+=term;}
}
fn accumulate2(sum:ptr<function,vec2<f32>>,error:ptr<function,vec2<f32>>,term:vec2<f32>){
  if(COMPENSATED){let next=(*sum)+term;(*error)+=select((term-next)+(*sum),((*sum)-next)+term,abs(*sum)>=abs(term));(*sum)=next;}
  else{(*sum)+=term;}
}
fn sp(x: f32) -> f32 {
  let e=exp(-abs(x));var tail=log(1.0+e);
  // WGSL has no log1p. Preserve the small negative tail in float32 rather
  // than losing it when 1+e rounds to one.
  if(e<0.01){tail=e*(1.0+e*(-0.5+e*(1.0/3.0-0.25*e)));}
  return max(x,0.0)+tail;
}
fn sigmoid(x: f32) -> f32 { let e=exp(-abs(x)); return select(e/(1.0+e),1.0/(1.0+e),x>=0.0); }
fn attention(frame:u32,s:u32)->f32 {
  var amin=areas[frame*32u]; for(var j=1u;j<S;j++){ amin=min(amin,areas[frame*32u+j]); }
  var denom=0.0; for(var j=0u;j<S;j++){ denom+=exp(amin-areas[frame*32u+j]); }
  return exp(amin-areas[frame*32u+s])/denom;
}
alias Mat = array<vec2<f32>,4>;
fn loadmat(o:u32)->Mat { var m:Mat;for(var j=0u;j<4u;j++){m[j]=cache[o+j];}return m; }
fn storemat(o:u32,m:Mat){for(var j=0u;j<4u;j++){cache[o+j]=m[j];}}
fn mm(a:Mat,b:Mat)->Mat {
  var result:Mat; for(var row=0u;row<2u;row++){for(var col=0u;col<2u;col++){
    result[row*2u+col]=mul(a[row*2u],b[col])+mul(a[row*2u+1u],b[2u+col]);
  }}return result;
}
fn mmadj(go:u32,ao:u32,bo:u32,gao:u32,gbo:u32){
  for(var row=0u;row<2u;row++){for(var col=0u;col<2u;col++){
    let g=cache[go+row*2u+col];for(var k=0u;k<2u;k++){
      let ai=row*2u+k;let bi=k*2u+col;
      cache[gao+ai]+=mul(g,conj(cache[bo+bi]));cache[gbo+bi]+=mul(g,conj(cache[ao+ai]));
    }
  }}
}
struct Section {k:Mat, dk:Mat};
fn section(a:f32,omega:f32)->Section {
  let li=LENGTH/32.0;let sa=2.0*PI*sqrt(a/PI)*li;
  let lii=RHO*li/(2.0*a);let ri=sa/(2.0*a*a)*sqrt(RHO*MU*0.5);let ci=a*li/(RHO*C*C);
  let lw=2.1/sa;let rw=800.0/sa;let cw=sa/84500.0;
  let za=vec2<f32>(ri*sqrt(omega),lii*omega);let dza=vec2<f32>(-1.5*za.x,-za.y);
  let b1=vec2<f32>(rw,lw*omega-1.0/(omega*cw));let wall=conj(b1)/dot(b1,b1);
  let y=wall+vec2<f32>(0.0,omega*ci);let dy=0.5*wall+vec2<f32>(0.0,omega*ci);
  let ay=mul(za,y);let a2=mul(za,za);let day=mul(dza,y)+mul(za,dy);let da2=2.0*mul(za,dza);
  var result:Section;
  result.k[0]=vec2<f32>(1.0,0.0)+ay;result.k[1]=-2.0*za-mul(a2,y);result.k[2]=-y;result.k[3]=result.k[0];
  result.dk[0]=day;result.dk[1]=-2.0*dza-mul(da2,y)-mul(a2,dy);result.dk[2]=-dy;result.dk[3]=day;return result;
}
fn radiation(area:f32,omega:f32)->array<vec2<f32>,2>{
  let lr=8.0*RHO/(3.0*PI*sqrt(PI*area));let rr=128.0*RHO*C/(9.0*PI*PI*area);let x=omega*lr;
  let z=vec2<f32>(rr*x*x,rr*rr*x)/(rr*rr+x*x);var r:array<vec2<f32>,2>;r[0]=z;r[1]=-mul(mul(z,z),vec2<f32>(1.0/rr,-0.5/x));return r;
}
alias Values = array<vec2<f32>,23>;
fn vals(f:Mat,b:Mat,z:vec2<f32>)->Values {
  var v:Values;for(var j=0u;j<4u;j++){v[j]=f[j];v[4u+j]=b[j];}v[8]=z;
  v[9]=-v[1];v[10]=div(v[9],v[3]);v[11]=mul(v[6],v[8]);v[12]=v[4]-v[11];
  v[13]=mul(v[7],v[8]);v[14]=v[13]-v[5];v[15]=div(v[14],v[12]);v[16]=v[10]+v[15];
  v[17]=mul(v[12],v[16]);v[18]=mul(v[8],v[10]);v[19]=div(v[18],v[17]);
  v[20]=mul(v[2],v[15]);v[21]=v[0]-v[20];v[22]=div(vec2<f32>(1.0,0.0),v[21]);return v;
}
fn am(v:ptr<function,Values>,g:ptr<function,Values>,y:u32,a:u32,b:u32){
  (*g)[a]+=mul((*g)[y],conj((*v)[b]));(*g)[b]+=mul((*g)[y],conj((*v)[a]));
}
fn ad(v:ptr<function,Values>,g:ptr<function,Values>,y:u32,a:u32,b:u32){
  let ga=div((*g)[y],conj((*v)[b]));(*g)[a]+=ga;(*g)[b]-=mul(ga,conj((*v)[y]));
}
fn aa(g:ptr<function,Values>,y:u32,a:u32,b:u32,sign:f32){(*g)[a]+=(*g)[y];(*g)[b]+=sign*(*g)[y];}

@compute @workgroup_size(64) fn pack_input(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<FRAMES*N){${compensatedInput ? 'framePairs[i]=vec4<f32>(uin[i],uinLow[i],0.0,0.0);' : 'frameA[i]=vec2<f32>(uin[i],0.0);'}}}
@compute @workgroup_size(64) fn transfer_forward(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=FRAMES*BINS){return;}let frame=i/BINS;let f=i%BINS;let omega=2.0*PI*max(1e-8,f32(f)*SR/f32(N));let base=i*S*4u;
  let zr=radiation(areas[frame*32u+31u],omega)[0];
  for(var s=0u;s<S;s++){storemat(base+s*4u,section(areas[frame*32u+s],omega).k);}
  storemat(BLOCK+base,loadmat(base));storemat(2u*BLOCK+base,loadmat(base+(S-1u)*4u));
  for(var s=1u;s<S;s++){
    storemat(BLOCK+base+s*4u,mm(loadmat(base+s*4u),loadmat(BLOCK+base+(s-1u)*4u)));
    storemat(2u*BLOCK+base+s*4u,mm(loadmat(2u*BLOCK+base+(s-1u)*4u),loadmat(base+(S-1u-s)*4u)));
  }
  for(var s=0u;s<S;s++){
    let v=vals(loadmat(BLOCK+base+s*4u),loadmat(2u*BLOCK+base+s*4u),zr);let j=(frame*S+s)*BINS+f;transfer[j]=v[22];transfer[HF+j]=v[19];
  }
  let last=loadmat(BLOCK+base+(S-1u)*4u);transfer[HT+i]=div(zr,last[0]-mul(last[2],zr));
}
@compute @workgroup_size(64) fn flow_spectrum(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=FRAMES*S*N){return;}let tube=i/N;let frame=tube/S;let k=i%N;let f=select(N-k,k,k<=N/2u);
  var z=mul(transfer[tube*BINS+f],U[frame*N+f]);if(k==0u||k==N/2u){z.y=0.0;}if(k>N/2u){z=conj(z);}tubeA[i]=z;
}
@compute @workgroup_size(64) fn gate_forward(@builtin(global_invocation_id) id:vec3<u32>){
  let tube=id.x;if(tube>=FRAMES*S){return;}let frame=tube/S;let s=tube%S;let a=areas[frame*32u+s];let attn=attention(frame,s);var prev=0.0;
  for(var t=0u;t<N;t++){let i=tube*N+t;prev+=(tubeB[i].x-prev)*ALPHA;lowpassed[i]=prev;
    let re2=RE_SCALE*prev*prev/a;tubeA[i]=vec2<f32>(1e-4*noise[i]*sp(re2-RE_THRESHOLD)*attn,0.0);}
}
@compute @workgroup_size(64) fn output_spectrum(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=FRAMES*N){return;}let frame=i/N;let k=i%N;let f=select(N-k,k,k<=N/2u);var z=mul(U[frame*N+f],transfer[HT+frame*BINS+f]);var sumError=vec2<f32>(0.0);
  for(var s=0u;s<S;s++){let tube=frame*S+s;accumulate2(&z,&sumError,mul(tubeB[tube*N+f],transfer[HF+tube*BINS+f]));}z+=sumError;
  if(k==0u||k==N/2u){z.y=0.0;}if(k>N/2u){z=conj(z);}frameA[i]=z;
}
@compute @workgroup_size(64) fn unpack_output(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<FRAMES*N){output[i]=frameB[i].x;}}
@compute @workgroup_size(64) fn pack_cotangent(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<FRAMES*N){${compensatedCotangent ? 'framePairs[i]=vec4<f32>(cotangent[i],cotangentLow[i],0.0,0.0);' : 'frameA[i]=vec2<f32>(cotangent[i],0.0);'}}}
@compute @workgroup_size(64) fn spectral_backward(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=FRAMES*S*N){return;}let tube=i/N;let frame=tube/S;let f=i%N;if(f>=BINS){tubeA[i]=vec2<f32>(0.0);return;}
  let endpoint=f==0u||f==N/2u;var g=frameB[frame*N+f]*(select(2.0,1.0,endpoint)/f32(N));if(endpoint){g.y=0.0;}
  let j=tube*BINS+f;gtransfer[HF+j]=mul(g,conj(tubeB[i]));tubeA[i]=mul(g,conj(transfer[HF+j]));
  if(tube%S==0u){gtransfer[HT+frame*BINS+f]=mul(g,conj(U[frame*N+f]));}
}
@compute @workgroup_size(64) fn gate_backward(@builtin(global_invocation_id) id:vec3<u32>){
  let tube=id.x;if(tube>=FRAMES*S){return;}let frame=tube/S;let s=tube%S;let a=areas[frame*32u+s];let attn=attention(frame,s);
  var carry=0.0;var totals=vec2<f32>(0.0);var sumError=vec2<f32>(0.0);
  for(var reverse=0u;reverse<N;reverse++){
    let t=N-1u-reverse;let i=tube*N+t;let lp=lowpassed[i];let re2=RE_SCALE*lp*lp/a;let x=re2-RE_THRESHOLD;
    let base=tubeB[i].x*f32(N)*1e-4*noise[i];let gx=base*attn*sigmoid(x);accumulate2(&totals,&sumError,vec2<f32>(-gx*re2,base*sp(x)));
    carry+=gx*2.0*RE_SCALE*lp/a;tubeA[i]=vec2<f32>(ALPHA*carry,0.0);carry*=1.0-ALPHA;
  }gategrad[tube]=totals+sumError;
}
@compute @workgroup_size(64) fn flow_backward(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=FRAMES*S*BINS){return;}let tube=i/BINS;let f=i%BINS;let frame=tube/S;let endpoint=f==0u||f==N/2u;
  var g=tubeB[tube*N+f]*(select(2.0,1.0,endpoint)/f32(N));if(endpoint){g.y=0.0;}gtransfer[i]=mul(g,conj(U[frame*N+f]));
}
@compute @workgroup_size(64) fn transfer_backward(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=FRAMES*BINS){return;}let frame=i/BINS;let f=i%BINS;let omega=2.0*PI*max(1e-8,f32(f)*SR/f32(N));let base=i*S*4u;
  let radiationValues=radiation(areas[frame*32u+31u],omega);let zr=radiationValues[0];var gz=vec2<f32>(0.0);
  for(var j=0u;j<S*4u;j++){cache[3u*BLOCK+base+j]=vec2<f32>(0.0);cache[4u*BLOCK+base+j]=vec2<f32>(0.0);cache[5u*BLOCK+base+j]=vec2<f32>(0.0);}
  for(var s=0u;s<S;s++){
    var v=vals(loadmat(BLOCK+base+s*4u),loadmat(2u*BLOCK+base+s*4u),zr);var g:Values;let j=(frame*S+s)*BINS+f;
    g[19]=gtransfer[HF+j];g[21]=mul(gtransfer[j],conj(-mul(v[22],v[22])));
    aa(&g,21u,0u,20u,-1.0);am(&v,&g,20u,2u,15u);ad(&v,&g,19u,18u,17u);am(&v,&g,18u,8u,10u);am(&v,&g,17u,12u,16u);
    aa(&g,16u,10u,15u,1.0);ad(&v,&g,15u,14u,12u);aa(&g,14u,13u,5u,-1.0);am(&v,&g,13u,7u,8u);
    aa(&g,12u,4u,11u,-1.0);am(&v,&g,11u,6u,8u);ad(&v,&g,10u,9u,3u);g[1]-=g[9];
    for(var q=0u;q<4u;q++){cache[4u*BLOCK+base+s*4u+q]+=g[q];cache[5u*BLOCK+base+s*4u+q]+=g[4u+q];}gz+=g[8];
  }
  let last=loadmat(BLOCK+base+(S-1u)*4u);let denominator=last[0]-mul(last[2],zr);let gn=div(gtransfer[HT+i],conj(denominator));let gd=-mul(gn,conj(transfer[HT+i]));
  gz+=gn-mul(gd,conj(last[2]));cache[4u*BLOCK+base+(S-1u)*4u]+=gd;cache[4u*BLOCK+base+(S-1u)*4u+2u]-=mul(gd,conj(zr));
  for(var reverse=0u;reverse<S-1u;reverse++){
    let s=S-1u-reverse;let j=S-1u-s;
    mmadj(4u*BLOCK+base+s*4u,base+s*4u,BLOCK+base+(s-1u)*4u,3u*BLOCK+base+s*4u,4u*BLOCK+base+(s-1u)*4u);
    mmadj(5u*BLOCK+base+s*4u,2u*BLOCK+base+(s-1u)*4u,base+j*4u,5u*BLOCK+base+(s-1u)*4u,3u*BLOCK+base+j*4u);
  }
  for(var q=0u;q<4u;q++){cache[3u*BLOCK+base+q]+=cache[4u*BLOCK+base+q];cache[3u*BLOCK+base+(S-1u)*4u+q]+=cache[5u*BLOCK+base+q];}
  for(var s=0u;s<S;s++){let dk=section(areas[frame*32u+s],omega).dk;var value=0.0;var sumError=0.0;for(var q=0u;q<4u;q++){accumulate(&value,&sumError,dot(cache[3u*BLOCK+base+s*4u+q],dk[q]));}freqgrad[i*32u+s]=value+sumError;}
  freqgrad[i*32u+31u]=dot(gz,radiationValues[1]);
}
@compute @workgroup_size(64) fn reduce_gradients(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=FRAMES*32u){return;}let frame=i/32u;let s=i%32u;var value=0.0;var sumError=0.0;
  for(var f=0u;f<BINS;f++){accumulate(&value,&sumError,freqgrad[(frame*BINS+f)*32u+s]);}
  if(s<S){var weighted=0.0;for(var j=0u;j<S;j++){weighted+=attention(frame,j)*gategrad[frame*S+j].y;}
    accumulate(&value,&sumError,gategrad[frame*S+s].x-areas[i]*attention(frame,s)*(gategrad[frame*S+s].y-weighted));}
  dlogareas[i]=value+sumError;
}
`;
  const module = device.createShaderModule({label: 'faithful acoustic forward and exact reverse', code: shader});
  const specs = {
    pack_input: [compensatedInput ? [1, 18, 19] : [1, 12], frames * n], transfer_forward: [[0, 6, 8], frames * bins], flow_spectrum: [[6, 14, 16], frames * sections * n],
    gate_forward: [[0, 2, 9, 14, 15], frames * sections], output_spectrum: [[6, 12, 15, 16], frames * n], unpack_output: [[3, 13], frames * n],
    pack_cotangent: [compensatedCotangent ? [4, 17, 18] : [4, 12], frames * n], spectral_backward: [[6, 7, 13, 14, 15, 16], frames * sections * n],
    gate_backward: [[0, 2, 9, 10, 14, 15], frames * sections], flow_backward: [[7, 15, 16], frames * sections * bins],
    transfer_backward: [[0, 6, 7, 8, 11], frames * bins], reduce_gradients: [[0, 5, 10, 11], frames * 32],
  };
  const kernels = {};
  let forwardFrames, inverseFrames, forwardTubes, inverseTubes, forwardPairedFrames;
  try {
    await Promise.all(Object.entries(specs).map(async ([name, [bindings, count]]) => {
      const pipeline = await device.createComputePipelineAsync({label: name, layout: 'auto', compute: {module, entryPoint: name}});
      const bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: bindings.map(binding => ({binding, resource: {buffer: resources[binding]}}))});
      kernels[name] = {pipeline, bindGroup, workgroups: Math.ceil(count / 64)};
    }));
    [forwardFrames, inverseFrames, forwardTubes, inverseTubes] = await Promise.all([
      createGpuFft(device, {...fftOptions, inputFormat: 'complex32', size: n, batchCount: frames}), createGpuFft(device, {...fftOptions, inputFormat: 'complex32', size: n, batchCount: frames, inverse: true}),
      createGpuFft(device, {...fftOptions, inputFormat: 'complex32', size: n, batchCount: frames * sections}), createGpuFft(device, {...fftOptions, inputFormat: 'complex32', size: n, batchCount: frames * sections, inverse: true}),
    ]);
    if (compensatedCotangent || compensatedInput) forwardPairedFrames = await createGpuFft(device, {...fftOptions, inputFormat: 'complex-pairs', size: n, batchCount: frames});
  } catch (error) { for (const b of buffers) b.destroy(); for (const f of [forwardFrames, inverseFrames, forwardTubes, inverseTubes, forwardPairedFrames]) f?.dispose?.(); throw error; }
  function dispatch(encoder, name) { const k = kernels[name], pass = encoder.beginComputePass({label: name}); pass.setPipeline(k.pipeline); pass.setBindGroup(0, k.bindGroup); pass.dispatchWorkgroups(k.workgroups); pass.end(); }
  function encodeForward(encoder) {
    dispatch(encoder, 'pack_input');
    if (compensatedInput) forwardPairedFrames.encode(encoder, framePairs, U);
    else forwardFrames.encode(encoder, frameA, U);
    dispatch(encoder, 'transfer_forward'); dispatch(encoder, 'flow_spectrum'); inverseTubes.encode(encoder, tubeA, tubeB);
    dispatch(encoder, 'gate_forward'); forwardTubes.encode(encoder, tubeA, tubeB);
    dispatch(encoder, 'output_spectrum'); inverseFrames.encode(encoder, frameA, frameB); dispatch(encoder, 'unpack_output');
  }
  function encodeBackward(encoder) {
    dispatch(encoder, 'pack_cotangent');
    if (compensatedCotangent) forwardPairedFrames.encode(encoder, framePairs, frameB);
    else forwardFrames.encode(encoder, frameA, frameB);
    dispatch(encoder, 'spectral_backward'); inverseTubes.encode(encoder, tubeA, tubeB);
    dispatch(encoder, 'gate_backward'); forwardTubes.encode(encoder, tubeA, tubeB); dispatch(encoder, 'flow_backward');
    dispatch(encoder, 'transfer_backward'); dispatch(encoder, 'reduce_gradients');
  }
  let disposed = false;
  function dispose() { if (disposed) return; disposed = true; for (const b of buffers) b.destroy(); for (const f of [forwardFrames, inverseFrames, forwardTubes, inverseTubes, forwardPairedFrames]) f?.dispose?.(); }
  const diagnosticBuffers = options.diagnostics ? Object.freeze({
    transferBuffer: transfer, transferCotangentBuffer: gtransfer,
    matrixCacheBuffer: cache, lowpassBuffer: lowpassed,
    noiseSpectrumBuffer: tubeB, sourceSpectrumBuffer: U,
    cotangentPairsBuffer: framePairs, cotangentSpectrumBuffer: frameB,
    frequencyGradientBuffer: frequencyGrad, gateGradientBuffer: gateGrad,
    // Complex storage is interleaved real/imaginary f32. Hc, Hfric are
    // [frame,31,bins]; Htract is [frame,bins], concatenated in that order.
    bins, sections, hfricComplexOffset: frames * sections * bins,
    htractComplexOffset: frames * sections * bins * 2,
  }) : undefined;
  return {frames, fftSize: n, areaBuffer, uinBuffer, uinLowBuffer, noiseBuffer, outputBuffer, cotangentBuffer, cotangentLowBuffer, dlogAreaBuffer, encodeForward, encodeBackward, dispose, diagnosticBuffers};
}
