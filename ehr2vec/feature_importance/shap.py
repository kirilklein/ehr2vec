import numpy as np
import torch
from typing import Dict
from ehr2vec.model.model import BertForFineTuning

class EHRMasker:
    """
    Masker for EHR data. Masks values in x with self.mask_value where mask is True.
    The __call__ method will be called by the explainer to mask the input data.
    """
    def __init__(self, vocabulary: dict) -> None:
        self.mask_value = vocabulary.get('[MASK]', None)
        if self.mask_value is None:
            raise ValueError('Mask value not found in vocabulary')
    
    def __call__(self, mask, x):
        """Mask values in x with self.mask_value where mask is True."""
        masked_x = np.where(mask, x, self.mask_value)
        return masked_x
    

class BEHRTWrapper(torch.nn.Module):
    """
    This wrapper is used to wrap the BEHRT model for SHAP explainer.
    The SHAP explainer will only mask the concept IDs, rest is passed unchanged to BEHRT.
    """
    def __init__(self, model: BertForFineTuning, batch: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.model = model
        self.batch = batch

    def __call__(self, concept: np.ndarray):
        """
        Compute the output of the model for the given concept IDs.
        To make compatible with SHAP, concepts are passed in the shape bs, 1, seq_len to shap explainer
        the explainer then passess n_permutations, seq_len to the model.
        We need to copy the other inputs to take the same shape as the concept.
        """
        batch_copy = self.batch.copy() # don't modify the original batch#
        concept = torch.from_numpy(concept) # shap explainer passes numpy array
        self.synchronise_shapes(batch_copy, concept)
        batch_copy['concept'] = concept
        output = self.model(batch=batch_copy).logits
        return output
    
    @staticmethod
    def synchronise_shapes(batch:Dict[str, torch.Tensor], concept: torch.Tensor)->None:
        """
        Synchronise the shape of the batch with the concept.
        """
        for key in batch: # copy all entries in batch to be the same shape along first dimension as concepts
            if key != 'concept': # this will be taken from the explainer and is already in the correct shape
                batch[key] = batch[key].repeat(concept.shape[0], 1)
