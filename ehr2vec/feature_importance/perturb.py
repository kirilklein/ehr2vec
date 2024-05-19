import torch
from transformers import BertModel

from ehr2vec.common.config import Config
from ehr2vec.embeddings.ehr import PerturbedEHREmbeddings


class PerturbationModel(torch.nn.Module):
    def __init__(self, bert_model:BertModel, cfg:Config):
        super().__init__()
        self.config = cfg

        self.lambda_ = self.config.get('lambda', .01)
        self.bert_model = bert_model
        self.K = bert_model.config.hidden_size
        self.freeze_bert()
        self.noise_simulator = GaussianNoise(bert_model, cfg)
        self.regularization_term = 1/(self.K*self.lambda_)
        
        self.embeddings_perturb = PerturbedEHREmbeddings(self.bert_model.config)
        self.embeddings_perturb.set_parameters(self.bert_model.embeddings)
        self.embeddings_perturb.freeze_embeddings()


    def forward(self, batch: dict):
        original_output = self.bert_model(batch=batch)  
        perturbed_embeddings = self.embeddings_perturb(batch, self.noise_simulator)
        perturbed_output = self.bert_model(batch, perturbed_embeddings)
        loss = self.perturbation_loss(original_output, perturbed_output, batch)
        outputs = ModelOutputs(logits=original_output.logits, perturbed_logits=perturbed_output.logits, loss=loss)
        return outputs

    def freeze_bert(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def perturbation_loss(self, original_output, perturbed_output, batch: dict)->torch.Tensor:
        """
        Calculate the perturbation loss as presented in eq. 7 in the paper:
        Towards a deep and unified understanding of deep neural models in NLP.
        https://proceedings.mlr.press/v97/guan19a.html
        Here we use logits directly as "hidden states".
        Args:
            original_output: Model output without perturbation
            perturbed_output: Model output with perturbation
            batch: Input batch, needed to access the correct sigmas
        """
        logits = original_output.logits
        perturbed_logits = perturbed_output.logits
        squared_diff = (logits - perturbed_logits)**2
        sigmas = self.noise_simulator.sigmas_embedding.weight
        concept_sigmas = sigmas[batch['concept']]
        
        first_term = -torch.log(concept_sigmas).sum()
        second_term = self.regularization_term*squared_diff/(logits.std()+1e-6) # Add epsilon to avoid division by zero
        loss = first_term + second_term
        return loss.mean()
    
    def save_sigmas(self, path:str)->None:
        torch.save(self.noise_simulator.sigmas_embedding.weight, path)

class GaussianNoise(torch.nn.Module):
    """Simulate Gaussian noise with trainable sigma to add to the embeddings"""
    def __init__(self, bert_model, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert_model = bert_model # BERT model
        self.initialize()

    def initialize(self):
        """Initialize the noise module with an embedding layer for sigmas."""
        num_concepts = len(self.bert_model.embeddings.concept_embeddings.weight.data)
        self.sigmas_embedding = torch.nn.Embedding(num_concepts, 1)
        self.sigmas_embedding.weight.data.fill_(1.0)  # Initialize all sigma values to 1

    def simulate_noise(self, concepts, embeddings: torch.Tensor)->torch.Tensor:
        """Simulate Gaussian noise using the sigmas"""
        concept_sigmas = self.sigmas_embedding(concepts).squeeze(-1)
        std_normal_noise = torch.randn_like(embeddings)
        scaled_noise = std_normal_noise * concept_sigmas.unsqueeze(-1)
        return scaled_noise

class ModelOutputs:
    def __init__(self, logits=None, perturbed_logits=None, loss=None):
        self.loss = loss
        self.logits = logits
        self.perturbed_logits = perturbed_logits