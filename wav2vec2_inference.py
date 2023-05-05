import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor, Wav2Vec2ProcessorWithLM
import json
from pyctcdecode import build_ctcdecoder

class Wave2Vec2Inference:
    def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if use_lm_if_possible:
            vocab_dict = json.load(open(f"{model_name}/vocab.json"))

            decoder = build_ctcdecoder(
                labels=list(vocab_dict.keys()),
                kenlm_model_path=f"{model_name}/language_model/3gram.bin",
                alpha=0.5,
                unigrams=open(f"{model_name}/language_model/unigrams.txt").read().splitlines()
            )
            tokenizer = Wav2Vec2CTCTokenizer(f"{model_name}/vocab.json", unk_token="[UNK]",
                                             pad_token="[PAD]",
                                             word_delimiter_token="|")
            tokenizer.add_special_tokens({'additional_special_tokens': ['[HES]', '[SPELL]']})

            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                         do_normalize=True, return_attention_mask=True)
            feature_extractor._processor_class = "Wav2Vec2ProcessorWithLM"
            self.processor = Wav2Vec2ProcessorWithLM(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                decoder=decoder
            )
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name, return_attention_mask=True)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        self.hotwords = hotwords
        self.use_lm_if_possible = use_lm_if_possible

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device),
                                attention_mask=inputs.attention_mask.to(self.device)).logits            

        if hasattr(self.processor, 'decoder') and self.use_lm_if_possible:
            transcription = \
                self.processor.decode(logits[0].cpu().numpy(),                                      
                                      hotwords=self.hotwords,
                                      #hotword_weight=self.hotword_weight,  
                                      output_word_offsets=True,                                      
                                   )                             
            confidence = transcription.lm_score / len(transcription.text.split(" "))
            transcription = transcription.text       
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            confidence = self.confidence_score(logits,predicted_ids)

        return transcription, confidence   

    def confidence_score(self, logits, predicted_ids):
        scores = torch.nn.functional.softmax(logits, dim=-1)                                                           
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id), 
            predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)
        return total_average

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    
if __name__ == "__main__":
    print("Model test")
    asr = Wave2Vec2Inference("checkpoint-115000", use_lm_if_possible=True, use_gpu=False)
    text = asr.file_to_text("untitled.wav")
    print(text)
