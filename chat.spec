# chat.spec
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

chromadb_datas, chromadb_binaries, chromadb_hiddenimports = collect_all('chromadb')
fitz_datas, fitz_binaries, fitz_hiddenimports = collect_all('fitz')
lc_datas, lc_binaries, lc_hiddenimports = collect_all('langchain_text_splitters')

a = Analysis(
    ['chat.py'],
    pathex=['.'],
    binaries=fitz_binaries + chromadb_binaries,
    datas=fitz_datas + chromadb_datas + lc_datas + [('setup_ollama.sh', '.')],
    hiddenimports=(
        fitz_hiddenimports + chromadb_hiddenimports + lc_hiddenimports + [
            'chromadb.db.impl.sqlite',
            'chromadb.segment.impl.vector.local_hnsw',
            'chromadb.segment.impl.metadata.sqlite',
            'onnxruntime',
            'tokenizers',
            'src.ingest',
            'src.query',
        ]
    ),
    hookspath=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas,
    name='chatbot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    onefile=True,
)