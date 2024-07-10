import ray
import time
import argparse
import numpy as np
from typing import Any, Dict, List
from vllm import LLM, SamplingParams
from packaging.version import Version
from langchain import PromptTemplate, FewShotPromptTemplate
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


assert Version(ray.version) >= Version("2.22.0"), "Ray version must be at least 2.22.0"
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_id = "Qwen/Qwen2-72B-Instruct"

start_time = time.time()

parser = argparse.ArgumentParser(description="Translate a large text dataset using LLM")
parser.add_argument("--input_file", type=str, help="Path to the input text file")
parser.add_argument("--source_lang", type=str, help="Source language (e.g., en, vi, ko, th)")
parser.add_argument("--target_lang", type=str, help="Target language (e.g., en, vi, ko, th)")
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Set tensor parallelism per instance.")
parser.add_argument("--num_instances", type=int, default=1, help="Set number of instances. Each instance will use tensor_parallel_size GPUs.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for translation (default: 32)")
args = parser.parse_args()


def language_from_id(id: str):
    if id == "th":
        return "Thai"
    elif id == "vi":
        return "Vietnamese"
    elif id == "ko":
        return "Korean"
    else:
        return "English"
        
example_envi = [
     {
        "source_sentence": "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each.",
        "translated_sentence": "Vào hôm thứ Hai, các nhà khoa học thuộc Trường Y Đại học Stanford đã công bố phát minh một dụng cụ chẩn đoán mới có thể phân loại tế bào: một con chíp nhỏ có thể sản xuất bằng máy in phun tiêu chuẩn với giá khoảng một xu Mỹ mỗi chiếc."
    }, {
        "source_sentence": "Lead researchers say this may bring early detection of cancer, tuberculosis, HIV and malaria to patients in low-income countries, where the survival rates for illnesses such as breast cancer can be half those of richer countries.",
        "translated_sentence": "Các nhà nghiên cứu chính nói rằng điều này có thể giúp phát hiện sớm bệnh ung thư, bệnh lao, HIV và bệnh sốt rét cho bệnh nhân ở các nước có thu nhập thấp, nơi mà tỷ lệ sống sót khi mắc phải những bệnh như ung thư vú có thể chỉ bằng một nửa tỷ lệ đó ở những nước giàu."
    }, {
        "source_sentence": "The JAS 39C Gripen crashed onto a runway at around 9:30 am local time (0230 UTC) and exploded, closing the airport to commercial flights.",
        "translated_sentence": "Chiếc JAS 39C Gripen đâm xuống đường băng vào khoảng 9:30 sáng giờ địa phương (0230 UTC) và nổ tung, khiến cho phi trường phải đóng cửa các chuyến bay thương mại."
    }
]

example_vien = []
example_kovi = []
example_viko = []
example_then = []
example_enth = []

examples = []

if args.source_lang == "en" and args.target_lang == "vi":
    examples = example_envi
elif args.source_lang == "vi" and args.target_lang == "en":
    examples = example_vien
elif args.source_lang == "ko" and args.target_lang == "vi":
    examples = example_kovi
elif args.source_lang == "vi" and args.target_lang == "ko":
    examples = example_viko
elif args.source_lang == "en" and args.target_lang == "th":
    examples = example_enth
elif args.source_lang == "th" and args.target_lang == "en":
    examples = example_then

example_template = """
source sentence: {source_sentence}
AI: {translated_sentence}
"""

example_prompt = PromptTemplate(
    input_variables=["source_sentence", "translated_sentence"],
    template=example_template
)

prefix = """You are a translator with vast knowledge of human languages. Please translate the following from {input_language} to {output_language}.
To ensure accurate translation, please strictly follow the guidelines:
- Use commonly used words in {output_language} conversations and maintain grammatical correctness.
- Please provide a literal translation, considering cultural nuances if necessary.
- Do not add any extra information, strictly.
- Strictly, return only the translated text, and no extra words or information should be included.
- Do not add apostrophes to the returned string.
- Your tone is polite.
- Put your translation in the first line, please pay attention to this
"""
# and the suffix our user input and output indicator
suffix = """
source sentence: {source_sentence}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["source_sentence", "input_language", "output_language"],
    example_separator="\n"
)

# Create a class to do batch inference.
class LLMPredictor:

    def init(self):
        # Create an LLM.
        self.llm = LLM(model=model_id,
                       tensor_parallel_size=args.tensor_parallel_size,
                       gpu_memory_utilization=0.99,
                       enforce_eager=True)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(temperature=0.6, 
                                 top_p=0.9,
                                 stop_token_ids=[self.tokenizer.eos_token_id, 
                                                 self.tokenizer.convert_tokens_to_ids("<|eot_id|>")])
        # self.sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

    def call(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["prompt"], self.sampling_params)
        prompt: List[str] = []
        generated_text: List[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            output_text = ' '.join([o.text for o in output.outputs]).strip()
            # output_text = output.outputs[0].text
            generated_text.append(output_text)
            # print(f"\n========")
            # print(output_text)
        return {
            "source_text": batch["text"],
            "prompt": prompt,
            "generated_text": generated_text,
        }

def make_prompt(source_lang, target_lang, text):
    prompt = few_shot_prompt_template.format(source_sentence=text, input_language=source_lang, output_language=target_lang)
    return prompt

def make_prompts(batch: Dict[str,str])-> Dict[str, str]:
    batch['prompt'] = [make_prompt(language_from_id(args.source_lang), language_from_id(args.target_lang), text) for text in batch['text']]
    return batch

# Iference
# For tensor_parallel_size > 1, we need to create placement groups for vLLM
# to use. Every actor has to have its own placement group.
def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * args.tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))
    
ds = ray.data.read_text(args.input_file)
prompt_ds = ds.map_batches(make_prompts)

resources_kwarg: Dict[str, Any] = {}
if args.tensor_parallel_size == 1:
    # For tensor_parallel_size == 1, we simply set num_gpus=1.
    resources_kwarg["num_gpus"] = 1
else:
    # Otherwise, we have to set num_gpus=0 and provide
    # a function that will create a placement group for
    # each instance.
    resources_kwarg["num_gpus"] = 0
    resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

# Apply batch inference for all input data.
result_ds = prompt_ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=args.num_instances,
    # Specify the batch size for inference.
    batch_size=args.batch_size,
    **resources_kwarg,
)

output_folder = f"{args.input_file}-{args.source_lang}{args.target_lang}"
result_ds.write_parquet(output_folder)

end_time = time.time()
print("Time taken:", (end_time - start_time)/60, "mins")