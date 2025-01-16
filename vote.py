import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.generation import GenerationConfig, GenerationMixin
from voting_utils import sampling, calculate_entropy, most_common_from_list
from pprint import pp

class LogtisVotingModel():

    def __init__(self, model_name_list, method='weighted', verbose=False, weight_list=None):

        self._model_list = []
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

            self._model_list.append(model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_list[0], trust_remote_code=True
        )

        self.generation_config = GenerationConfig.from_pretrained(
            model_name_list[0], trust_remote_code=True
        )

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.generation_config.pad_token_id = pad_token_id

        self.device = self._model_list[0].device

        self.method = method.lower()
        assert self.method in ["entropy", "weighted"], NotImplementedError
        if weight_list is not None:
            assert len(weight_list) == len(model_name_list)
            self._weight_list = torch.tensor(weight_list, device=self.device)
        else:
            self._weight_list = torch.ones((len(model_name_list), ), device=self.device)

    
    @torch.no_grad()
    def generate(self, input_text, **kwargs):

        self.generation_config.update(**kwargs)        

        if isinstance(input_text, list):
            #First generation, apply template
            inputs = self.tokenizer.apply_chat_template(input_text, return_tensors='pt', return_dict=True).to(self.device)
            cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=self.device)
            generated = inputs.input_ids.clone()  # Initialize with the input sequence
        else:
            encode_result = self.tokenizer(input_text, return_tensors='pt', return_attention_mask=True)
            inputs = {"input_ids": encode_result.input_ids.to(self.device), "attention_mask": encode_result.attention_mask.to(self.device)}
            cache_position = torch.arange(encode_result.input_ids.shape[1], dtype=torch.int64, device=self.device)
            generated = encode_result.input_ids.clone()  # Initialize with the input sequence

        generated = generated.to(self.device)
        raw_length = generated.size(-1)
        batch_size = generated.size(0)

        # Iterate until max_length or all sequences finish
        if self.generation_config.max_new_tokens is None:
            n_iterates = self.generation_config.max_length - raw_length
        else:
            n_iterates = self.generation_config.max_new_tokens

        past_key_value_list = [DynamicCache() for _ in range(len(self._model_list))]
        for _ in range(n_iterates):
            # Step 1: Forward pass
            logits_list = None
            for idx, model in enumerate(self._model_list):
                outputs = model(**inputs, 
                                past_key_values=past_key_value_list[idx], 
                                use_cache=True,
                                cache_position=cache_position)
                if self.generation_config.use_cache:
                    past_key_value_list[idx] = outputs.past_key_values
                # logits: (B, L, n_vocab) -> (B, n_vocab), B = 0 in all test
                # Using the last pos as next token
                logits = outputs.logits[:, -1, :]  
                # assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf values"
                logits *= self._weight_list[idx]
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
                next_token_ids = sampling(logits, self.generation_config.top_k, self.generation_config.top_p)
            else:
                next_token_ids = torch.argmax(logits, dim=1)

            # Step 5: Append next token to the sequence
            generated = torch.cat((generated, next_token_ids.unsqueeze(0)), dim=1)

            # Step 6: Stop if all sequences reach the EOS token
            if self.generation_config.eos_token_id is not None:
                if next_token_ids.item() in self.generation_config.eos_token_id:
                    break
            
            # Step 7: Prepare next forward call
            attention_mask = inputs["attention_mask"]
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            inputs = {"input_ids": next_token_ids.unsqueeze(0), "attention_mask": attention_mask}
            cache_position = cache_position[-1:] + 1 # add one more position for the next token

        return generated[:, raw_length:]
                

class TokenVotingModel():

    def __init__(self, model_name_list, verbose=False, eos_tokens=['']):
        self._model_list = []
        self._tokenizer_list = []
        self._generation_config_list = []
        for _, model_name in enumerate(model_name_list):
            if verbose:
                print(f"Loading {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", trust_remote_code=True
            ).eval()
            self._model_list.append(model)
        
            generation_config = GenerationConfig.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._generation_config_list.append(generation_config)

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._tokenizer_list.append(tokenizer)

        self.device = self._model_list[0].device
        self.generation_config = self._generation_config_list[0]
        self.eos_token_list = ["<|endoftext|>", "<|eot_id|>"]

        self.eos_token_list.extend(eos_tokens)
        self.counter = [0] * len(model_name_list)

    
    @torch.no_grad()
    def generate(self, input_text, **kwargs):

        self.generation_config.update(**kwargs)        
        # Iterate until max_length or all sequences finish
        n_iterates = self.generation_config.max_new_tokens if self.generation_config.max_new_tokens is not None else 20

        past_key_value_list = [DynamicCache() for _ in range(len(self._model_list))]
        cache_position_list = [None] * len(self._model_list)
        input_list = [None] * len(self._model_list)
        generated_text = ''
        for n in range(n_iterates):
            next_token_list = []
            for idx, (model, tokenizer) in enumerate(zip(self._model_list, self._tokenizer_list)):
                # Step 1: Encode text into tokens
                if n == 0:
                    if isinstance(input_text, list):
                    #First generation, apply template
                        input_list[idx] = tokenizer.apply_chat_template(input_text, return_tensors='pt', return_dict=True).to(self.device)
                        cache_position_list[idx] = torch.arange(input_list[idx].input_ids.shape[1], dtype=torch.int64, device=self.device)
                    else:
                        encode_result = tokenizer(input_text, return_tensors='pt', return_attention_mask=True)
                        input_list[idx] = {"input_ids": encode_result.input_ids.to(self.device), "attention_mask": encode_result.attention_mask.to(self.device)}
                        cache_position_list[idx] = torch.arange(encode_result.input_ids.shape[1], dtype=torch.int64, device=self.device)


                # Step 2: Forward pass
                outputs = model(**input_list[idx], 
                                past_key_values=past_key_value_list[idx], 
                                use_cache=True,
                                cache_position=cache_position_list[idx])
                # if self.generation_config.use_cache:
                #     past_key_value_list[idx] = outputs.past_key_values

                # logits: (B, L, n_vocab) -> (B, n_vocab)
                # Using the last pos as next token
                logits = outputs.logits[:, -1, :]  
                # assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf values"

                # Step 3: Get next token of current model
                next_token_id = torch.argmax(logits, dim=1)
                
                # Step 4: Decode next token to get plain text
                next_token_list.append(tokenizer.decode(next_token_id))

            # Step 5: Choose one token from list of results
            chosen_next_token = most_common_from_list(next_token_list)
            # interlude: count where the valid opinion come from
            for i, token in enumerate(next_token_list): 
                if token == chosen_next_token:
                    self.counter[i] += 1 

            # Step 6: Append next token to the sequence
            generated_text += chosen_next_token

            # Step 7: Stop if all sequences reach the EOS token
            if chosen_next_token in self.eos_token_list:
                break

            # Step 8: Prepare next forward call
            for idx, tokenizer, in enumerate(self._tokenizer_list):
                attention_mask = input_list[idx]["attention_mask"]
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                # Add constraints to limit only one new token ids, or the kv cache would panic
                next_token_ids = tokenizer.encode(chosen_next_token, return_tensors='pt', add_special_tokens=False, max_length=1, truncation="longest_first").to(self.device)
                input_list[idx] = {"input_ids": next_token_ids, "attention_mask": attention_mask}
                cache_position_list[idx] = cache_position_list[idx][-1:] + 1
        
        return generated_text

class TokenVotingSingleModel():
    """
    Using single model for token ensemble, require different input prompt
    """
    def __init__(self, model_name, n_model, verbose=False, eos_tokens=['']):
        self.n_model = n_model
        if verbose:
            print(f"Loading {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        ).eval()
    
        self.generation_config = GenerationConfig.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.device = self.model.device
        self.eos_token_list = ["<|endoftext|>", "<|eot_id|>"]

        self.eos_token_list.extend(eos_tokens)

    
    @torch.no_grad()
    def generate_with_different_input(self, input_text_list, **kwargs):
        """
        Each model accepts different input
        """
        self.generation_config.update(**kwargs)        
        # Iterate until max_length or all sequences finish
        n_iterates = self.generation_config.max_new_tokens if self.generation_config.max_new_tokens is not None else 20

        past_key_value_list = [DynamicCache() for _ in range(self.n_model)]
        cache_position_list = [None] * self.n_model
        input_list = [None] * self.n_model
        generated_text = ''
        for n in range(n_iterates):
            next_token_list = []
            for idx in range(self.n_model):
                # Step 1: Encode text into tokens
                if n == 0:
                    if isinstance(input_text_list[idx], list):
                    #First generation, apply template
                        input_list[idx] = self.tokenizer.apply_chat_template(input_text_list[idx], return_tensors='pt', return_dict=True).to(self.device)
                        cache_position_list[idx] = torch.arange(input_list[idx].input_ids.shape[1], dtype=torch.int64, device=self.device)
                    else:
                        encode_result = self.tokenizer(input_text_list[idx], return_tensors='pt', return_attention_mask=True)
                        input_list[idx] = {"input_ids": encode_result.input_ids.to(self.device), "attention_mask": encode_result.attention_mask.to(self.device)}
                        cache_position_list[idx] = torch.arange(encode_result.input_ids.shape[1], dtype=torch.int64, device=self.device)


                # Step 2: Forward pass
                outputs = self.model(**input_list[idx], 
                                past_key_values=past_key_value_list[idx], 
                                use_cache=True,
                                cache_position=cache_position_list[idx])
                # if self.generation_config.use_cache:
                #     past_key_value_list[idx] = outputs.past_key_values

                # logits: (B, L, n_vocab) -> (B, n_vocab)
                # Using the last pos as next token
                logits = outputs.logits[:, -1, :]  
                # assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf values"

                # Step 3: Get next token of current model
                next_token_id = torch.argmax(logits, dim=1)
                
                # Step 4: Decode next token to get plain text
                next_token_list.append(self.tokenizer.decode(next_token_id))

            # Step 5: Choose one token from list of results
            chosen_next_token = most_common_from_list(next_token_list)
            # chosen_next_token = next_token_list[0]

            # Step 6: Append next token to the sequence
            generated_text += chosen_next_token

            # Step 7: Stop if all sequences reach the EOS token
            if chosen_next_token in self.eos_token_list:
                break

            # Step 8: Prepare next forward call
            for idx in range(self.n_model):
                attention_mask = input_list[idx]["attention_mask"]
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
                # Add constraints to limit only one new token ids, or the kv cache would panic
                next_token_ids = self.tokenizer.encode(chosen_next_token, return_tensors='pt', add_special_tokens=False, max_length=1, truncation="longest_first").to(self.device)
                input_list[idx] = {"input_ids": next_token_ids, "attention_mask": attention_mask}
                cache_position_list[idx] = cache_position_list[idx][-1:] + 1
        
        return generated_text