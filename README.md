# Emotional Tutor

### Overview

An emotional tutor designed for children, with the purpose of tutoring a child by answering any relevant questions, and acting out emotions, since there are fewer tools for chilrdren.

![Introduction Image](images/intro_image.png)



## Setup

```bash
pip install -r requirements.txt
```

## Quickstart
```bash
pip install -r requirements.txt
python main.py

```
**Requirements:** Python 3.10+, Hume API key, Deepseek API key

## Structure
- main.py (The main code)
- It follows the following pipeline (Speech input → ASR → LLM → emotional TTS → spoken answer):

![Pipeline Image](images/pipeline.png)
