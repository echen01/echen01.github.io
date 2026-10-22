//! Stationary voiced subset of diffTLM's frequency-domain acoustic model.
//!
//! Equations follow echen01/diffTLM, difftlm/tract.py at
//! b05116e06afbd7a569f3b9ac1d03a52c9d18efed (MIT license).
//! As in that implementation, N area values include a separate lip boundary:
//! the first N-1 sections propagate sound, each with length L/N.
//! This educational browser experiment has no turbulence, temporal geometry,
//! glottal inference, or neural field. Its gradients are reverse-mode derivatives
//! of the actual complex transmission-line model, never finite differences.

use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub const N: usize = 32;
const MAX_FREQ: usize = 4097;
const RHO: f64 = 1.14e-3;
const C: f64 = 3.43e4;
const MU: f64 = 1.86e-4;
const MW: f64 = 2.1;
const BW: f64 = 800.0;
const KW: f64 = 84500.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct Complex { pub re: f64, pub im: f64 }
impl Complex {
    fn new(re: f64, im: f64) -> Self { Self { re, im } }
    fn norm_sqr(self) -> f64 { self.re * self.re + self.im * self.im }
}
impl Add for Complex { type Output = Self; fn add(self, b: Self) -> Self { Self::new(self.re+b.re,self.im+b.im) } }
impl Sub for Complex { type Output = Self; fn sub(self, b: Self) -> Self { Self::new(self.re-b.re,self.im-b.im) } }
impl Neg for Complex { type Output = Self; fn neg(self) -> Self { Self::new(-self.re,-self.im) } }
impl Mul for Complex { type Output = Self; fn mul(self, b: Self) -> Self { Self::new(self.re*b.re-self.im*b.im,self.re*b.im+self.im*b.re) } }
impl Div for Complex { type Output = Self; fn div(self, b: Self) -> Self { let d=b.norm_sqr(); Self::new((self.re*b.re+self.im*b.im)/d,(self.im*b.re-self.re*b.im)/d) } }
impl Mul<f64> for Complex { type Output = Self; fn mul(self, b:f64) -> Self { Self::new(self.re*b,self.im*b) } }

/// Lip pressure / glottal volume-velocity response, in the source CGS units.
pub fn transfer(areas: &[f64; N], length: f64, frequency: f64) -> Complex {
    let omega = 2.0 * PI * frequency.max(1e-8);
    let li = length / N as f64;
    let lip_area=areas[N-1];
    let lr=8.0*RHO/(3.0*PI*(PI*lip_area).sqrt());
    let rr=128.0*RHO*C/(9.0*PI*PI*lip_area);
    let zr=Complex::new(0.0,rr*omega*lr)/Complex::new(rr,omega*lr);
    let one=Complex::new(1.0,0.0);
    // Only the first column of the product is needed by H_tract.
    let (mut k00,mut k10)=(one,Complex::default());
    for &a in &areas[..N-1] {
        let sa=2.0*(PI*a).sqrt()*li;
        let inductance=RHO*li/(2.0*a);
        let resistance=sa/(2.0*a*a)*(RHO*MU*0.5).sqrt();
        let capacitance=a*li/(RHO*C*C);
        let za=Complex::new(resistance*omega.sqrt(),omega*inductance);
        let zb1=Complex::new(BW/sa,omega*MW/sa-KW/(omega*sa));
        let zb2=Complex::new(0.0,-1.0/(omega*capacitance));
        let zb=zb1*zb2/(zb1+zb2);
        let t00=one+za/zb;
        let t01=-za*2.0-za*za/zb;
        let t10=-one/zb;
        (k00,k10)=(t00*k00+t01*k10,t10*k00+t00*k10);
    }
    zr/(k00-k10*zr)
}

// A small scalar reverse-mode tape. Complex arithmetic is decomposed into real
// operations, so its backward pass obeys real multivariate calculus directly.
type Var=usize;
const NONE:usize=usize::MAX;
struct Node { value:f64, adjoint:f64, a:Var, b:Var, da:f64, db:f64 }
struct Tape { nodes:Vec<Node> }
impl Tape {
    fn new() -> Self { Self { nodes:Vec::with_capacity(5000) } }
    fn node(&mut self,value:f64,a:Var,b:Var,da:f64,db:f64)->Var { let i=self.nodes.len();self.nodes.push(Node{value,adjoint:0.0,a,b,da,db});i }
    fn constant(&mut self,v:f64)->Var {self.node(v,NONE,NONE,0.0,0.0)}
    fn value(&self,a:Var)->f64 {self.nodes[a].value}
    fn add(&mut self,a:Var,b:Var)->Var {self.node(self.value(a)+self.value(b),a,b,1.0,1.0)}
    fn sub(&mut self,a:Var,b:Var)->Var {self.node(self.value(a)-self.value(b),a,b,1.0,-1.0)}
    fn mul(&mut self,a:Var,b:Var)->Var {self.node(self.value(a)*self.value(b),a,b,self.value(b),self.value(a))}
    fn div(&mut self,a:Var,b:Var)->Var {let bv=self.value(b);self.node(self.value(a)/bv,a,b,1.0/bv,-self.value(a)/(bv*bv))}
    fn scale(&mut self,a:Var,b:f64)->Var {self.node(self.value(a)*b,a,NONE,b,0.0)}
    fn sqrt(&mut self,a:Var)->Var {let v=self.value(a).sqrt();self.node(v,a,NONE,0.5/v,0.0)}
    fn exp(&mut self,a:Var)->Var {let v=self.value(a).exp();self.node(v,a,NONE,v,0.0)}
    fn ln(&mut self,a:Var)->Var {self.node(self.value(a).ln(),a,NONE,1.0/self.value(a),0.0)}
    fn backward(&mut self,out:Var) {
        self.nodes[out].adjoint=1.0;
        for i in (0..=out).rev() {
            let n=&self.nodes[i];let (a,b,da,db,g)=(n.a,n.b,n.da,n.db,n.adjoint);
            if a!=NONE {self.nodes[a].adjoint+=g*da;}
            if b!=NONE {self.nodes[b].adjoint+=g*db;}
        }
    }
}
#[derive(Clone,Copy)]
struct Z {r:Var,i:Var}
fn z_const(t:&mut Tape,r:f64,i:f64)->Z {Z{r:t.constant(r),i:t.constant(i)}}
fn z_add(t:&mut Tape,a:Z,b:Z)->Z {Z{r:t.add(a.r,b.r),i:t.add(a.i,b.i)}}
fn z_sub(t:&mut Tape,a:Z,b:Z)->Z {Z{r:t.sub(a.r,b.r),i:t.sub(a.i,b.i)}}
fn z_scale(t:&mut Tape,a:Z,s:f64)->Z {Z{r:t.scale(a.r,s),i:t.scale(a.i,s)}}
fn z_mul(t:&mut Tape,a:Z,b:Z)->Z {
    let rr=t.mul(a.r,b.r);let ii=t.mul(a.i,b.i);let ri=t.mul(a.r,b.i);let ir=t.mul(a.i,b.r);
    Z{r:t.sub(rr,ii),i:t.add(ri,ir)}
}
fn z_div(t:&mut Tape,a:Z,b:Z)->Z {
    let rr=t.mul(b.r,b.r);let ii=t.mul(b.i,b.i);let den=t.add(rr,ii);
    let rr=t.mul(a.r,b.r);let ii=t.mul(a.i,b.i);let ir=t.mul(a.i,b.r);let ri=t.mul(a.r,b.i);
    let nr=t.add(rr,ii);let ni=t.sub(ir,ri);Z{r:t.div(nr,den),i:t.div(ni,den)}
}

fn log_transfer_with_tape(t:&mut Tape,inputs:&[Var;N],length:f64,frequency:f64)->Var {
    let omega=2.0*PI*frequency.max(1e-8);let li=length/N as f64;
    let mut areas=[0;N];for (i,&input) in inputs.iter().enumerate(){areas[i]=t.exp(input);}
    let lip=areas[N-1];
    let sqrt_a=t.sqrt(lip);let lr_const=t.constant(8.0*RHO/(3.0*PI*PI.sqrt()));
    let lr=t.div(lr_const,sqrt_a);let rr_const=t.constant(128.0*RHO*C/(9.0*PI*PI));let rr=t.div(rr_const,lip);
    let om_lr=t.scale(lr,omega);let num=t.mul(rr,om_lr);
    let numerator=Z{r:t.constant(0.0),i:num};let denominator=Z{r:rr,i:om_lr};let zr=z_div(t,numerator,denominator);
    let one=z_const(t,1.0,0.0);let zero=t.constant(0.0);let mut k00=one;let mut k10=z_const(t,0.0,0.0);
    for &a in &areas[..N-1] {
        let sqrt_a=t.sqrt(a);let sa=t.scale(sqrt_a,2.0*PI.sqrt()*li);
        let a2=t.mul(a,a);let sa_over_a2=t.div(sa,a2);let ri=t.scale(sa_over_a2,0.5*(RHO*MU*0.5).sqrt()*omega.sqrt());
        let lc=t.constant(omega*RHO*li/2.0);let induct=t.div(lc,a);let za=Z{r:ri,i:induct};
        let bw=t.constant(BW);let wall_r=t.div(bw,sa);
        let wall_i_const=t.constant(omega*MW-KW/omega);let wall_i=t.div(wall_i_const,sa);let zb1=Z{r:wall_r,i:wall_i};
        let air_i_const=t.constant(-RHO*C*C/(omega*li));let air_i=t.div(air_i_const,a);let zb2=Z{r:zero,i:air_i};
        let zb_n=z_mul(t,zb1,zb2);let zb_d=z_add(t,zb1,zb2);let zb=z_div(t,zb_n,zb_d);
        let ratio=z_div(t,za,zb);let t00=z_add(t,one,ratio);
        let za2=z_mul(t,za,za);let za2_zb=z_div(t,za2,zb);let minus2za=z_scale(t,za,-2.0);let t01=z_sub(t,minus2za,za2_zb);
        let inv_zb=z_div(t,one,zb);let t10=z_scale(t,inv_zb,-1.0);
        let aa=z_mul(t,t00,k00);let bc=z_mul(t,t01,k10);let new00=z_add(t,aa,bc);
        let ca=z_mul(t,t10,k00);let dc=z_mul(t,t00,k10);let new10=z_add(t,ca,dc);
        k00=new00;k10=new10;
    }
    let kz=z_mul(t,k10,zr);let den=z_sub(t,k00,kz);let h=z_div(t,zr,den);
    let re2=t.mul(h.r,h.r);let im2=t.mul(h.i,h.i);let power=t.add(re2,im2);let lp=t.ln(power);t.scale(lp,0.5)
}

/// Mean squared log-magnitude loss and its reverse-mode gradient w.r.t. log A.
/// The target is supplied in dB to share an API with the displayed spectrum.
pub fn loss_and_gradient(areas:&[f64;N],target_db:&[f64],frequencies:&[f64],length:f64)->(f64,[f64;N]) {
    assert_eq!(target_db.len(),frequencies.len());
    let mut loss=0.0;let mut gradient=[0.0;N];let mut tape=Tape::new();
    let scale=1.0/frequencies.len() as f64;
    for (&frequency,&target) in frequencies.iter().zip(target_db) {
        tape.nodes.clear();
        let mut inputs=[0;N];for (i,&area) in areas.iter().enumerate(){inputs[i]=tape.constant(area.ln());}
        let h=log_transfer_with_tape(&mut tape,&inputs,length,frequency);
        let target=tape.constant(target*std::f64::consts::LN_10/20.0);
        let residual=tape.sub(h,target);let sq=tape.mul(residual,residual);let objective=tape.scale(sq,scale);
        loss+=tape.value(objective);tape.backward(objective);
        for (i,&input) in inputs.iter().enumerate(){gradient[i]+=tape.nodes[input].adjoint;}
    }
    (loss,gradient)
}

// Fixed exchange buffers keep the browser boundary dependency-free. Each Worker
// owns its own WebAssembly instance, and JS recreates views after memory growth.
static mut AREAS:[f64;N]=[1.0;N];
static mut FREQUENCIES:[f64;MAX_FREQ]=[0.0;MAX_FREQ];
static mut TARGET:[f64;MAX_FREQ]=[0.0;MAX_FREQ];
static mut REAL:[f64;MAX_FREQ]=[0.0;MAX_FREQ];
static mut IMAG:[f64;MAX_FREQ]=[0.0;MAX_FREQ];
static mut DB:[f64;MAX_FREQ]=[0.0;MAX_FREQ];
static mut GRADIENT:[f64;N]=[0.0;N];

#[no_mangle] pub extern "C" fn areas_ptr()->*mut f64 {std::ptr::addr_of_mut!(AREAS).cast()}
#[no_mangle] pub extern "C" fn frequencies_ptr()->*mut f64 {std::ptr::addr_of_mut!(FREQUENCIES).cast()}
#[no_mangle] pub extern "C" fn target_ptr()->*mut f64 {std::ptr::addr_of_mut!(TARGET).cast()}
#[no_mangle] pub extern "C" fn real_ptr()->*mut f64 {std::ptr::addr_of_mut!(REAL).cast()}
#[no_mangle] pub extern "C" fn imag_ptr()->*mut f64 {std::ptr::addr_of_mut!(IMAG).cast()}
#[no_mangle] pub extern "C" fn db_ptr()->*mut f64 {std::ptr::addr_of_mut!(DB).cast()}
#[no_mangle] pub extern "C" fn gradient_ptr()->*mut f64 {std::ptr::addr_of_mut!(GRADIENT).cast()}

#[no_mangle] pub unsafe extern "C" fn evaluate_response(length:f64,n:usize)->i32 {
    if n==0||n>MAX_FREQ||!length.is_finite()||length<=0.0 {return -1;}
    let areas=AREAS;if areas.iter().any(|a| !a.is_finite()||*a<=0.0){return -2;}
    for i in 0..n {
        if !FREQUENCIES[i].is_finite()||FREQUENCIES[i]<0.0{return -3;}
        let h=transfer(&areas,length,FREQUENCIES[i]);REAL[i]=h.re;IMAG[i]=h.im;DB[i]=10.0*h.norm_sqr().log10();
    }
    0
}
#[no_mangle] pub unsafe extern "C" fn evaluate_loss(length:f64,n:usize)->f64 {
    if n==0||n>MAX_FREQ||!length.is_finite()||length<=0.0 {return f64::NAN;}
    let areas=AREAS;if areas.iter().any(|a| !a.is_finite()||*a<=0.0){return f64::NAN;}
    let frequencies=std::slice::from_raw_parts(frequencies_ptr(),n);let target=std::slice::from_raw_parts(target_ptr(),n);
    let (loss,gradient)=loss_and_gradient(&areas,target,frequencies,length);GRADIENT=gradient;loss
}

#[cfg(test)]
mod tests {
    use super::*;
    fn areas()->[f64;N] {std::array::from_fn(|i|1.8+0.7*(i as f64*0.37).sin())}
    #[test] fn reverse_gradient_matches_centered_finite_difference() {
        let a=areas();let target_a=[2.7;N];let frequencies=[80.0,203.0,503.0,1067.0,2331.0,3893.0,4979.0];
        let target:Vec<_>=frequencies.iter().map(|&f|10.0*transfer(&target_a,17.5,f).norm_sqr().log10()).collect();
        let (_,gradient)=loss_and_gradient(&a,&target,&frequencies,17.5);
        let objective=|x:&[f64;N]|frequencies.iter().zip(&target).map(|(&f,&y)|{let ln=0.5*transfer(x,17.5,f).norm_sqr().ln();(ln-y*std::f64::consts::LN_10/20.0).powi(2)}).sum::<f64>()/frequencies.len() as f64;
        for i in 0..N {let mut ap=a;let mut am=a;let eps=1e-5;ap[i]*=f64::exp(eps);am[i]*=f64::exp(-eps);let numeric=(objective(&ap)-objective(&am))/(2.0*eps);assert!((gradient[i]-numeric).abs()<2e-6*(1.0+numeric.abs()),"index {i}: AD={} FD={numeric}",gradient[i]);}
    }
    #[test] fn identical_target_has_zero_loss_and_gradient() {
        let a=areas();let f=[50.0,100.0,1000.0,3000.0,7000.0];let target:Vec<_>=f.iter().map(|&f|10.0*transfer(&a,17.5,f).norm_sqr().log10()).collect();let (loss,grad)=loss_and_gradient(&a,&target,&f,17.5);
        assert!(loss<1e-20);assert!(grad.iter().all(|g|g.abs()<1e-10));
    }
    #[test] fn finite_across_interactive_area_range() {for area in [0.12,0.5,1.0,8.0,16.0] {for frequency in [0.0,80.0,500.0,3000.0,8000.0,16000.0] {let h=transfer(&[area;N],17.5,frequency);assert!(h.re.is_finite()&&h.im.is_finite());}}}
}
