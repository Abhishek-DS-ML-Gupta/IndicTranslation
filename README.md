# 🌍 Indic Document Translation Pipeline

A powerful **AI-based multilingual document translator** built using **IndicTrans2**, capable of translating **DOCX, PDF, and TXT files** across 20+ Indian languages while preserving structure.

---

## ✨ Features

* 🔥 Translate between **English ↔ Indic** and **Indic ↔ Indic**
* 📄 Supports:

  * `.docx` (with formatting preservation)
  * `.pdf` (text extraction mode)
  * `.txt`
* 🧠 Uses **IndicTrans2 (1B models)** for high-quality translation
* ⚡ **GPU accelerated (CUDA required)**
* 💾 Memory-optimized batch translation (prevents OOM)
* 📊 Preserves:

  * Paragraphs
  * Tables
  * Layout (DOCX)
* 🎯 Multi-language support (20+ Indian languages)

---

## 🧠 Supported Languages

English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu, Odia, Assamese, Sanskrit, Kashmiri, Sindhi, Manipuri, Santali, Nepali, Konkani, Dogri, Bodo, Maithili

---

## 🏗️ Tech Stack

* Python 🐍
* PyTorch 🔥
* HuggingFace Transformers 🤗
* IndicTransToolkit 🇮🇳
* Gradio 🎨 (UI)
* pdfplumber 📄
* python-docx 📝

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/indic-document-translator.git
cd indic-document-translator
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate   # Linux
myenv\Scripts\activate      # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python app.py
```

Then open in browser:

```
http://localhost:7860
```

---

## 🖥️ Requirements

* ✅ Python ≥ 3.8
* ✅ NVIDIA GPU (Required)
* ✅ CUDA Installed
* ❌ CPU not supported (due to large model size)

---

## 📂 Project Structure

```
├── app.py
├── requirements.txt
├── README.md
├── gradio_temp/
├── uploads/
└── outputs/
```

---

## ⚡ Performance Tips

* Reduce batch size if facing OOM:

  ```python
  batch_size = 4
  ```
* Use GPU with higher VRAM (16GB+ recommended)
* Close other GPU processes:

  ```bash
  nvidia-smi
  ```

---

## ⚠️ Limitations

* PDF layout is not preserved (text only)
* Large documents may take time
* Requires GPU (1B models are heavy)

---

## 🔥 Future Improvements

* OCR support (image → text → translation)
* Layout-preserving PDF translation
* Multi-GPU parallel processing
* REST API (FastAPI)
* Drag & Drop UI enhancements

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Abhishek Gupta**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
