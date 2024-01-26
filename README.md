# GPT-2 Fine-tuning

### Transfer Learning with GPT-2 on Stanford Encyclopedia of Philosophy

This project utilizes transfer learning techniques to fine-tune the GPT-2 model on articles from the Stanford Encyclopedia of Philosophy. The fine-tuning process is based on the GPT-2 implementation provided by `hugfaceguy0001/stanford_plato` available on Hugging Face.

For more details on the GPT-2 model, visit: [https://huggingface.co/gpt2](https://huggingface.co/gpt2)

## Quick Start

Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

Run the fine-tuning script:
if you want to download the tokenized text download. if not run the preprocessing.py
```bash
python preprocessing.py
python reTrain.py
```


## Acknowledgments

- **Stanford Encyclopedia of Philosophy**: For providing a comprehensive and high-quality dataset.
- **Hugging Face**: For their open-source GPT-2 implementation and the platform to share models.
- **hugfaceguy0001/stanford_plato**: For the fine-tuned model and inspiration.
