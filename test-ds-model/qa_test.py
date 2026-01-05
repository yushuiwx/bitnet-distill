import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
import json
from datetime import datetime
import torch.nn as nn

def replace_linear_with_fusedbit(model, standard_bitnet = False, inquant_layers_keys = [], weight_quant_method = 'minmax'):
    from bitnetsrc import core

    for name, module in model.named_modules():
        if 'lm_head' in name: # we always keep the precision of last lm_head  
            continue

        # skip params in inquant_layers_keys
        quant = True
        for key in inquant_layers_keys:
            if key in name:
                quant = False
        if not quant:
            continue

        if isinstance(module, nn.Linear):
            if standard_bitnet:
                if 'o_proj' in name or 'down_proj' in name:
                    fusedbit_layer = core.BitLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        should_rms = True,
                        bias=(module.bias is not None),
                        weight_quant_method=weight_quant_method
                    )
                else:
                    fusedbit_layer = core.BitLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        should_rms = False,
                        bias=(module.bias is not None),
                        weight_quant_method=weight_quant_method
                    )
            else:
                fusedbit_layer = core.BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    should_rms = True,
                    bias=(module.bias is not None),
                    weight_quant_method=weight_quant_method
                )

            # Copy existing weights/bias
            with torch.no_grad():
                fusedbit_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    fusedbit_layer.bias.copy_(module.bias)

            # Replace the original module in its parent
            parent_path = name.rsplit('.', 1)
            if len(parent_path) == 1:
                setattr(model, parent_path[0], fusedbit_layer)
            else:
                parent_module_name, child_name = parent_path
                parent_module = dict(model.named_modules())[parent_module_name]
                setattr(parent_module, child_name, fusedbit_layer)
    return model


def chat_with_model(model_name, questions, device="cuda" if torch.cuda.is_available() else "cpu", save_path=None):
    print(f"==> Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32,
                                                 device_map=None, trust_remote_code=True).to("cuda:0") 
    model = replace_linear_with_fusedbit(model)
    model = model.to("cuda:0") 
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    results = []
    print(f"\n==> Starting QA on device: {device}\n")

    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                                tokenize=False, add_generation_prompt=True)

        output = pipe(prompt, max_new_tokens=1024, temperature=0.7, do_sample=True)
        answer = output[0]["generated_text"][len(prompt):].strip()
        print(f"A{i}: {answer}\n")

        results.append({"question": question, "answer": answer})

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"==> Results saved to {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QA test for Hugging Face transformers models.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path (e.g., 'Qwen/Qwen2.5-1.5B')")
    parser.add_argument("--questions", type=str, nargs="+", required=False,
                        default=["Which hormone acts as a lipophilic hormone on nuclear receptors, differentiating it from hydrophilic hormones that act on cytosolic receptors?", "Which drugs from the list Fexofenadine, Phenytoin, Carbamazepine, Azithromycin, and Penicillin are known to affect CYP 3A4 enzymes?", "A 17-year-old girl presents to the emergency department with a painful scalp rash, excessive thirst, frequent urination, and a history of headaches. Physical examination reveals a disfiguring red rash on the scalp and radiographs show lytic skull lesions. What specific finding is most likely to be revealed on electron microscopy examination of this patient's condition?"],
                        help="List of questions for testing.")
    parser.add_argument("--save", action="store_true", help="Whether to save results to a JSON file.")
    args = parser.parse_args()

    save_path = None
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"qa_results_{args.model_name.replace('/', '_')}_{timestamp}.json"

    chat_with_model(args.model_name, args.questions, save_path=save_path)
