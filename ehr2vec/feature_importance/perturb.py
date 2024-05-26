import torch
from transformers import BertModel

from ehr2vec.common.config import Config
from ehr2vec.embeddings.ehr import PerturbedEHREmbeddings


class PerturbationModel(torch.nn.Module):
    def __init__(self, bert_model:BertModel, cfg:Config, concept_frequency=None):
        """Lambda determines how much the """
        super().__init__()
        self.config = cfg

        self.bert_model = bert_model
        self.freeze_bert()
        self.noise_simulator = GaussianNoise(bert_model, cfg)
        
        self.lambda_ = self.config.get('lambda', .01)
        self.K = bert_model.config.hidden_size
        regularization_term = 1/(self.K*self.lambda_)
        self.register_buffer('regularization_term', torch.tensor(regularization_term))
        
        inverse_frequency = self.set_inverse_frequency(concept_frequency)
    
        self.register_buffer('sqrt_inverse_frequency', torch.sqrt(inverse_frequency))

        self.embeddings_perturb = PerturbedEHREmbeddings(self.bert_model.config)
        self.embeddings_perturb.set_parameters(self.bert_model.embeddings)
        self.embeddings_perturb.freeze_embeddings()
        
    def set_inverse_frequency(self, concept_frequency):
        """Set the inverse frequency of the concepts to the sigmas. If not set, all sigmas are set to 1.0."""
        if concept_frequency is not None:
            if len(concept_frequency)!=len(self.noise_simulator.sigmas_embedding.weight):
                raise ValueError("Concept frequency should have the same length as the sigmas.")
            return (1/(concept_frequency+1e-6))
        return torch.ones_like(self.noise_simulator.sigmas_embedding.weight)

    def forward(self, batch: dict):
        original_output = self.bert_model(batch=batch,output_hidden_states=True)  
        perturbed_embeddings = self.embeddings_perturb(batch, self.noise_simulator)
        perturbed_output = self.bert_model(batch, perturbed_embeddings, output_hidden_states=True)
        loss = self.perturbation_loss(original_output, perturbed_output, batch)
        outputs = ModelOutputs(logits=original_output.logits, perturbed_logits=perturbed_output.logits, loss=loss,
                               hidden_states=original_output.hidden_states, perturbed_hidden_states=perturbed_output.hidden_states)
        return outputs

    def freeze_bert(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def perturbation_loss(self, original_output, perturbed_output, batch)->torch.Tensor:
        """
        Calculate the perturbation loss as presented in eq. 7 in the paper:
        Towards a deep and unified understanding of deep neural models in NLP.
        https://proceedings.mlr.press/v97/guan19a.html
        Calculate the perturbation loss, focusing on hidden states for a better alignment with mutual information principles.
        Args:
            original_output: Model output without perturbation
            perturbed_output: Model output with perturbation
            batch: Input batch, needed to access the correct sigmas
        """
        # Assuming hidden states are accessible from original_output and perturbed_output
        original_hidden = original_output.hidden_states[-1]  # Last layer hidden states
        perturbed_hidden = perturbed_output.hidden_states[-1]
        
        # Calculate squared differences in hidden states
        squared_diff = (original_hidden - perturbed_hidden) ** 2

        sigmas = self.noise_simulator.sigmas_embedding.weight
        
        # Regularization term on sigmas without harsh scaling
        first_term = -torch.log(sigmas).nanmean()

        # Normalize squared differences
        second_term = (self.regularization_term * squared_diff / (original_hidden.std() + 1e-6)).mean()

        # Combine the terms to form the final loss
        loss = first_term + second_term
        return loss

    def log(self, logger):
        log_string = "Perturbation model:\n"
        log_string += f"\t Regularization term: {self.regularization_term}\n"
        log_string += f"\tMin sigma: {self.noise_simulator.sigmas_embedding.weight.min()}\n"
        log_string += f"\tMax sigma: {self.noise_simulator.sigmas_embedding.weight.max()}\n"
        log_string += f"\tMean sigma: {self.noise_simulator.sigmas_embedding.weight.mean()}\n"
        logger.info(log_string)
        
    def save_sigmas(self, path:str)->None:
        torch.save(self.noise_simulator.sigmas_embedding.weight.flatten(), path)

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
        self.sigmas_embedding.weight.data.fill_(1e-3)  # Initialize all sigma values to 1

    def simulate_noise(self, concepts, embeddings: torch.Tensor)->torch.Tensor:
        """Simulate Gaussian noise using the sigmas"""
        concept_sigmas = self.sigmas_embedding(concepts).squeeze(-1)
        std_normal_noise = torch.randn_like(embeddings, device=embeddings.device)
        scaled_noise = std_normal_noise * concept_sigmas.unsqueeze(-1)
        return scaled_noise

class ModelOutputs:
    def __init__(self, logits=None, perturbed_logits=None, loss=None, hidden_states=None, perturbed_hidden_states=None):
        self.loss = loss
        self.logits = logits
        self.perturbed_logits = perturbed_logits
        self.hidden_states = hidden_states
        self.perturbed_hidden_states = perturbed_hidden_states