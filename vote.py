import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig, GenerationMixin
from voting_utils import sampling, calculate_entropy
from pprint import pp

class LogtisVotingModel():

    def __init__(self, model_name_list, method='weighted', verbose=False, weight_list=None):

        self.model_list = []
        n_vocab = None
        for idx, model_name in enumerate(model_name_list):
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", trust_remote_code=True
            ).eval()

            # pp(self.config)
            if verbose:
                print(f"{model_name} Loaded, n_vocab:{model.config.vocab_size}")
            if n_vocab:
                assert n_vocab == model.config.vocab_size, "vocab size should be the same"
            else:
                n_vocab = model.config.vocab_size

            self.model_list.append(model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_list[0], trust_remote_code=True
        )

        self.generation_config = GenerationConfig.from_pretrained(
            model_name_list[0], trust_remote_code=True
        )

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.generation_config.pad_token_id = pad_token_id

        self.device = self.model_list[0].device

        self.method = method.lower()
        assert self.method in ["entropy", "weighted"], NotImplementedError
        if weight_list is not None:
            assert len(weight_list) == len(model_name_list)
            self.weight_list = torch.tensor(weight_list, device=self.device)
        else:
            self.weight_list = torch.ones((len(model_name_list), ), device=self.device)

    
    @torch.no_grad()
    def generate(self, input_ids, **kwargs):

        batch_size = input_ids.size(0)
        generated = input_ids.clone()  # Initialize with the input sequence

        self.generation_config.update(**kwargs)        

        # Iterate until max_length or all sequences finish
        if self.generation_config.max_new_tokens is None:
            n_iterates = self.generation_config.max_length - input_ids.size(-1)
        else:
            n_iterates = self.generation_config.max_new_tokens
        for _ in range(n_iterates):
            # Step 1: Forward pass
            logits_list = None
            past_key_value_list = [None] * len(self.model_list)
            for idx, model in enumerate(self.model_list):
                outputs = model(input_ids=generated, 
                                past_key_values=past_key_value_list[idx], 
                                use_cache=self.generation_config.use_cache)
                if self.generation_config.use_cache:
                    past_key_value_list[idx] = outputs.past_key_values
                # logits: (B, L, n_vocab) -> (B, n_vocab), B = 0 in all test
                # Using the last pos as next token
                logits = outputs.logits[:, -1, :]  
                # assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf values"
                logits *= self.weight_list[idx]
                if self.method == "entropy":
                    # Higher Entropy -> higher uncertainty -> less weight
                    logits *= 1.0 / (calculate_entropy(logits) + 1)

                logits_list = torch.concat([logits_list, logits], dim=0) if logits_list is not None else logits
            
            logits = torch.mean(logits_list, dim=0, keepdim=True)


            # Step 2: Apply repetition penalty
            if self.generation_config.repetition_penalty:
                for i in range(batch_size):
                    logits[i, generated[i, -1]] /= self.generation_config.repetition_penalty

            # Step 3: Apply temperature
            if self.generation_config.temperature:
                logits = logits / self.generation_config.temperature

            # Step 4: Get next token
            if self.generation_config.do_sample:
                next_token = sampling(logits, self.generation_config.top_k, self.generation_config.top_p)
            else:
                next_token = torch.argmax(logits, dim=1)

            # Step 5: Append next token to the sequence
            generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)

            # Step 6: Stop if all sequences reach the EOS token
            if self.generation_config.eos_token_id is not None:
                if next_token.item() in self.generation_config.eos_token_id:
                    break

        return generated
                
        