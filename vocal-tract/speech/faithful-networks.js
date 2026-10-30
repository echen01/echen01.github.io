import {createFaithfulField, loadFaithfulInitialization, validateFaithfulHiddenDim} from './faithful-field.js?v=fixed-grid';
import {createFaithfulMfnField, loadFaithfulMfnInitialization} from './faithful-mfn-field.js?v=mfn-area-scale-1';

// Architecture and optimizer settings from difftlm.fields and fit_sample.py.
export function faithfulNetwork(fieldType = 'rff', hiddenDim = fieldType === 'mfn' ? 256 : 512) {
  if (fieldType === 'mfn') {
    if (hiddenDim !== 256) throw new Error('The MFN example uses the original 256-wide FourierNet.');
    return {label: 'MFN', peak: .005, load: loadFaithfulMfnInitialization, create: createFaithfulMfnField};
  }
  if (fieldType !== 'rff') throw new Error('Choose RFF or MFN for the neural field.');
  validateFaithfulHiddenDim(hiddenDim);
  return {label: 'RFF', peak: .001, load: loadFaithfulInitialization, create: createFaithfulField};
}
