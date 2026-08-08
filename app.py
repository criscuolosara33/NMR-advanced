import streamlit as st
from streamlit_ketcher import st_ketcher
import requests
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import warnings

st.set_page_config(page_title="NMR Simulator", layout="wide")

# --- CSS E COSTANTI ---
BORDEAUX = '#6B1422'
BORDEAUX_HOVER = '#822433'

st.markdown(f"""
<style>
    html, body, .stMarkdown, .stText, h1, h2, h3, h4, table {{ font-family: 'Palatino', serif !important; }}
    div.stButton > button:first-child {{ background-color: {BORDEAUX}; color: white; border: none; }}
    div.stButton > button:hover {{ background-color: {BORDEAUX_HOVER}; color: white; }}
</style>
""", unsafe_allow_html=True)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
plt.rcParams['font.family'] = 'serif'

# --- GESTIONE STATO ---
if 'smiles_corrente' not in st.session_state: st.session_state.smiles_corrente = ""
if 'parametri_attivi' not in st.session_state: st.session_state.parametri_attivi = False

# --- FUNZIONI CHIMICHE (Invariate) ---
def calcola_proprieta(mol):
    mol_h = Chem.AddHs(mol)
    n_tetra = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [6, 14]) 
    n_tri = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [7, 15]) 
    n_mono = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [1, 9, 17, 35, 53]) 
    dbe = n_tetra + 1 - (n_mono / 2.0) + (n_tri / 2.0)
    mw = Descriptors.MolWt(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol_h)
    return {'formula': formula, 'mw': mw, 'dbe': dbe, 'mol_h': mol_h, 'mol_no_h': mol}

# --- UI MAIN ---
st.title("NMR Simulator")

# Ketcher deve essere sempre richiamato per non perdere la molecola
smiles = st_ketcher()

if smiles:
    st.session_state.smiles_corrente = smiles
    
    if st.session_state.mostra_parametri:
        st.markdown("### Impostazioni Acquisizione")
        c1, c2 = st.columns(2)
        freq_1h = c1.selectbox("Frequenza 1H (MHz)", [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0], index=2)
        solv_1h = c2.selectbox("Solvente 1H", ["CDCl3", "DMSO-d6", "D2O", "CD3OD"])
        c3, c4, c5 = st.columns(3)
        freq_13c = c3.selectbox("Frequenza 13C (MHz)", [75.0, 100.0, 125.0, 150.0, 200.0, 250.0], index=2)
        solv_13c = c4.selectbox("Solvente 13C", ["CDCl3", "DMSO-d6", "D2O", "CD3OD"])
        modo_13c = c5.selectbox("Tecnica 13C", ["Broadband", "DEPT-135", "DEPT-90", "APT"])
        
        col1, col2 = st.columns(2)
        if col1.button("Esegui 1H-NMR"):
            st.session_state.tipo = '1h'
            st.session_state.p = {'freq': freq_1h, 'solv': solv_1h}
            st.session_state.mostra_parametri = False
            st.rerun()
        if col2.button("Esegui 13C-NMR"):
            st.session_state.tipo = '13c'
            st.session_state.p = {'freq': freq_13c, 'solv': solv_13c, 'modo': modo_13c}
            st.session_state.mostra_parametri = False
            st.rerun()
    else:
        if st.button("Modifica Parametri"):
            st.session_state.mostra_parametri = True
            st.rerun()
            
        # Logica di calcolo sicura
        if st.session_state.tipo:
            mol = Chem.MolFromSmiles(st.session_state.smiles_corrente)
            if mol:
                # Esegui calcoli e visualizzazione solo se la molecola è valida
                props = calcola_proprieta(mol)
                st.write("Calcolo in corso...")
                # ... (Qui inserisci la logica di calcolo/plotting precedente) ...
            else:
                st.error("Molecola non valida.")
else:
    st.info("Disegna una molecola per iniziare.")
