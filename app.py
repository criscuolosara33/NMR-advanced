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

# --- CSS E COSTANTI ESTETICHE ---
BORDEAUX = '#6B1422'
BORDEAUX_HOVER = '#822433'

st.markdown(f"""
<style>
    /* Tipografia Palatino Globale */
    html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, table, th, td {{
        font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif !important;
    }}
    
    div[data-testid="metric-container"] {{
        background-color: #fafafa;
        border: 1px solid #e6e6e6;
        padding: 15px 20px;
        border-radius: 6px;
        box-shadow: 1px 2px 4px rgba(0,0,0,0.04);
    }}
    
    div.stButton > button:first-child {{ 
        background-color: {BORDEAUX}; 
        color: white; 
        border: none;
        border-radius: 4px;
        font-weight: bold;
        letter-spacing: 0.5px;
        transition: all 0.2s ease-in-out;
    }}
    div.stButton > button:hover {{ 
        background-color: {BORDEAUX_HOVER}; 
        color: white; 
        box-shadow: 0 4px 6px rgba(107, 20, 34, 0.2);
    }}
    
    hr {{ margin-top: 1.5em; margin-bottom: 1.5em; border-color: #e6e6e6; }}
</style>
""", unsafe_allow_html=True)

# --- CONFIGURAZIONE MATPLOTLIB PER IL PDF ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Times New Roman', 'serif']

# --- INIZIALIZZAZIONE STATO ROBUSTA ---
if 'ultimo_smiles' not in st.session_state:
    st.session_state.ultimo_smiles = ""
if 'stato_app' not in st.session_state:
    st.session_state.stato_app = "input" 
if 'parametri' not in st.session_state:
    st.session_state.parametri = {}

# --- FUNZIONI CHIMICHE ---
def calcola_proprieta(mol):
    mol_h = Chem.AddHs(mol)
    n_tetra = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [6, 14]) 
    n_tri = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [7, 15]) 
    n_mono = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [1, 9, 17, 35, 53]) 
    
    dbe = n_tetra + 1 - (n_mono / 2.0) + (n_tri / 2.0)
    mw = Descriptors.MolWt(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol_h)
    
    formula_dbe_str = rf"n_{{IV}} + 1 - \frac{{n_{{I}}}}{{2}} + \frac{{n_{{III}}}}{{2}}"
    formula_dbe_val_str = rf"{n_tetra} + 1 - \frac{{{n_mono}}}{{2}} + \frac{{{n_tri}}}{{2}}"
    
    return {'formula': formula, 'mw': mw, 'dbe': dbe, 'formula_dbe_str': formula_dbe_str, 'formula_dbe_val_str': formula_dbe_val_str, 'mol_h': mol_h, 'mol_no_h': mol}

def ottieni_nomi_pubchem(smiles):
    try:
        url_iupac = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/property/IUPACName/JSON"
        res_iupac = requests.get(url_iupac, timeout=5)
        iupac_name = res_iupac.json()['PropertyTable']['Properties'][0]['IUPACName'] if res_iupac.status_code == 200 else "N/D"
        
        url_syn = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/synonyms/JSON"
        res_syn = requests.get(url_syn, timeout=5)
        common_name = res_syn.json()['InformationList']['Information'][0]['Synonym'][0] if res_syn.status_code == 200 else "N/D"
        return iupac_name, common_name
    except Exception:
        return "Errore connessione", "Errore connessione"

def analizza_stereochimica(mol):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    commenti = []
    
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if chiral_centers:
        dettagli_chirali = []
        for idx, config in chiral_centers:
            if config in ['R', 'S']:
                dettagli_chirali.append(f"C{idx + 1} ({config})")
            else:
                dettagli_chirali.append(f"C{idx + 1} (Stereochimica non definita nel disegno)")
        
        commenti.append(f"Centri stereogenici: Rilevati agli atomi {', '.join(dettagli_chirali)}.")
        
        ch2_diastereotopici = [str(atom.GetIdx() + 1) for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 2]
        if ch2_diastereotopici:
            commenti.append(f"Protoni diastereotopici: I gruppi CH2 (atomi {', '.join(ch2_diastereotopici)}) sono in intorno chirale. Essendo diastereotopici, non sono chimicamente equivalenti (anisocroni) e presentano accoppiamento geminale (sistema AB o ABX).")
    else:
        commenti.append("Molecola achirale: Nessun centro stereogenico. I gruppi CH2 presentano protoni enantiotopici, chimicamente isocroni in ambienti achirali.")
    
    dettagli_ez = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo == Chem.BondStereo.STEREOE:
                a1, a2 = bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1
                dettagli_ez.append(f"C{a1}=C{a2} (E)")
            elif stereo == Chem.BondStereo.STEREOZ:
                a1, a2 = bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1
                dettagli_ez.append(f"C{a1}=C{a2} (Z)")
    
    if dettagli_ez:
        commenti.append(f"Isomeria geometrica: Rilevati doppi legami con configurazione {', '.join(dettagli_ez)}.")
        
    return commenti

def analisi_accoppiamenti_avanzati(mol):
    commenti = []
    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        commenti.append("Equivalenza Magnetica (Sistemi Aromatici): I protoni isocroni in para- o orto- disostituzioni simmetriche raramente sono magneticamente equivalenti, presentando J diverse verso gli altri nuclei. Generano sistemi complessi (es. AA'BB').")
        commenti.append("Accoppiamenti a Lungo Raggio: Nei sistemi aromatici sono attivi accoppiamenti orto (3J ≈ 7-9 Hz), meta (4J ≈ 2-3 Hz) e para (5J < 1 Hz).")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CH2]-[CH2]-[CH2]")):
        commenti.append("Accoppiamento Virtuale: Rilevata catena alifatica. I CH2 interni presentano una differenza di chemical shift minima rispetto alla J. Il collasso delle molteplicità di prim'ordine genera multipletti non interpretabili con la regola n+1.")
    if not commenti:
        commenti.append("I sistemi di spin alifatici semplici seguono l'approssimazione del prim'ordine. Valida la regola n+1.")
    return commenti

def descrivi_accoppiamento(mult):
    return {'s': "Singoletto.", 'd': "Doppietto.", 't': "Tripletto.", 'q': "Quartetto.", 'dd': "Doppio doppietto.", 'm': "Multipletto complesso.", 'br s': "Singoletto allargato."}.get(mult.lower(), "Segnale non risolto.")

def genera_picchi(center_ppm, mult_type, integral, freq):
    j_std, j_ortho, j_meta = 7.5/freq, 8.0/freq, 2.0/freq
    mult = mult_type.lower() if mult_type else 's'
    if mult == 'd': off, rat = [-j_std/2, j_std/2], [0.5, 0.5]
    elif mult == 't': off, rat = [-j_std, 0, j_std], [0.25, 0.5, 0.25]
    elif mult == 'q': off, rat = [-1.5*j_std, -0.5*j_std, 0.5*j_std, 1.5*j_std], [0.125, 0.375, 0.375, 0.125]
    elif mult == 'dd': off, rat = [-j_ortho/2 - j_meta/2, -j_ortho/2 + j_meta/2, j_ortho/2 - j_meta/2, j_ortho/2 + j_meta/2], [0.25, 0.25, 0.25, 0.25]
    elif mult == 'm': off, rat = np.linspace(-1.5*j_std, 1.5*j_std, 5), [0.1, 0.25, 0.3, 0.25, 0.1]
    else: off, rat = [0.0], [1.0]
    return [(center_ppm + o, r * integral) for o, r in zip(off, rat)]

def stima_locale_1h(mol_h):
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    groups = {}
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            rank = ranks[atom.GetIdx()]
            if rank not in groups: groups[rank] = []
            groups[rank].append(atom)

    signals, shifts_visti = [], []
    for rank, h_atoms in groups.items():
        rep_h = h_atoms[0]
        integral = len(h_atoms)
        neighbor = rep_h.GetNeighbors()[0]
        c_indices = {h.GetNeighbors()[0].GetIdx() + 1 for h in h_atoms}

        if neighbor.GetIsAromatic(): shift = 7.3
        elif neighbor.GetAtomicNum() == 8: shift = 4.5
        elif neighbor.GetAtomicNum() == 7: shift = 2.5
        elif neighbor.GetAtomicNum() == 16: shift = 1.5
        elif neighbor.GetAtomicNum() == 6:
            if neighbor.GetHybridization() == Chem.HybridizationType.SP2:
                is_aldehyde = any(bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(neighbor).GetAtomicNum() == 8 for bond in neighbor.GetBonds())
                shift = 9.8 if is_aldehyde else 5.5
            elif neighbor.GetHybridization() == Chem.HybridizationType.SP: shift = 2.8
            else: shift = 0.9 + (0.3 * sum(1 for a in neighbor.GetNeighbors() if a.GetAtomicNum() == 6))
        else: shift = 2.0

        while any(abs(shift - sv) < 0.05 for sv in shifts_visti): shift += 0.1
        shifts_visti.append(shift)

        vicini_h = sum(1 for c_neigh in neighbor.GetNeighbors() if c_neigh.GetAtomicNum() == 6 for h_atom in c_neigh.GetNeighbors() if h_atom.GetAtomicNum() == 1 and ranks[h_atom.GetIdx()] != rank) if neighbor.GetAtomicNum() == 6 else 0
        is_exch = neighbor.GetAtomicNum() in [7, 8, 16]
        mult = 'br s' if is_exch else ({0:'s', 1:'d', 2:'t', 3:'q', 4:'m', 5:'m', 6:'m'}.get(vicini_h, 'm') if neighbor.GetAtomicNum() == 6 else 's')

        signals.append({'delta': shift, 'multiplicity': mult, 'integral': integral, 'atoms': list(c_indices), 'is_exchangeable': is_exch, 'coupling_comment': descrivi_accoppiamento(mult)})
    return signals

def stima_locale_13c(mol_no_h):
    ranks = list(Chem.CanonicalRankAtoms(mol_no_h, breakTies=False))
    groups = {}
    for atom in mol_no_h.GetAtoms():
        if atom.GetAtomicNum() == 6:
            rank = ranks[atom.GetIdx()]
            if rank not in groups: groups[rank] = []
            groups[rank].append(atom)

    signals, shifts_visti = [], []
    for rank, c_atoms in groups.items():
        rep_c = c_atoms[0]
        integral = len(c_atoms)
        n_h_attached = rep_c.GetTotalNumHs()
        
        shift = 30.0
        n_neighbors_C = sum(1 for n in rep_c.GetNeighbors() if n.GetAtomicNum() == 6)
        n_neighbors_O = sum(1 for n in rep_c.GetNeighbors() if n.GetAtomicNum() == 8)
        n_neighbors_N = sum(1 for n in rep_c.GetNeighbors() if n.GetAtomicNum() == 7)

        if rep_c.GetHybridization() == Chem.HybridizationType.SP2:
            if rep_c.GetIsAromatic(): shift = 130.0
            elif any(mol_no_h.GetBondBetweenAtoms(rep_c.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE and n.GetAtomicNum() == 8 for n in rep_c.GetNeighbors()): shift = 170.0
            else: shift = 120.0
        elif rep_c.GetHybridization() == Chem.HybridizationType.SP: shift = 70.0
        else: shift += (n_neighbors_C * 8) + (n_neighbors_O * 40) + (n_neighbors_N * 20)

        while any(abs(shift - sv) < 0.5 for sv in shifts_visti): shift += 0.5
        shifts_visti.append(shift)
        
        tipo_c = "Cq" if n_h_attached == 0 else f"CH{n_h_attached}" if n_h_attached > 1 else "CH"
        signals.append({'delta': shift, 'multiplicity': 's', 'integral': integral, 'atoms': [atom.GetIdx() + 1 for atom in c_atoms], 'n_h': n_h_attached, 'tipo_c': tipo_c, 'is_exchangeable': False, 'coupling_comment': f"Singolo disaccoppiato ({tipo_c})"})
    return signals

def salva_pagina_uniforme(pdf, fig):
    fig.set_size_inches(11.69, 8.27) 
    pdf.savefig(fig, orientation='landscape', bbox_inches='tight')
    plt.close(fig)

# --- UI MAIN ---
st.title("NMR Simulator")

smiles = st_ketcher()

if smiles != st.session_state.ultimo_smiles:
    st.session_state.ultimo_smiles = smiles
    st.session_state.stato_app = "input"

if st.session_state.stato_app == "input":
    st.markdown("### Parametri 1H-NMR")
    c1, c2 = st.columns(2)
    freq_1h = c1.selectbox("Frequenza 1H (MHz)", [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0], index=2)
    solv_1h = c2.selectbox("Solvente 1H", ["CDCl3", "DMSO-d6", "D2O", "CD3OD"])
    
    st.markdown("### Parametri 13C-NMR")
    c3, c4, c5 = st.columns(3)
    freq_13c = c3.selectbox("Frequenza 13C (MHz)", [75.0, 100.0, 125.0, 150.0, 200.0, 250.0], index=2)
    solv_13c = c4.selectbox("Solvente 13C", ["CDCl3", "DMSO-d6", "D2O", "CD3OD"])
    modo_13c = c5.selectbox("Tecnica 13C", ["Broadband", "DEPT-135", "DEPT-90", "APT"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    cb1, cb2 = st.columns(2)
    
    if cb1.button("Acquisisci 1H", use_container_width=True):
        if not smiles:
            st.error("Errore: Nessuna molecola disegnata.")
        else:
            st.session_state.parametri = {'freq': freq_1h, 'solvente': solv_1h, 'tech': '1h'}
            st.session_state.stato_app = "calcolo_1h"
            st.rerun()
            
    if cb2.button("Acquisisci 13C", use_container_width=True):
        if not smiles:
            st.error("Errore: Nessuna molecola disegnata.")
        else:
            st.session_state.parametri = {'freq': freq_13c, 'solvente': solv_13c, 'tech': modo_13c}
            st.session_state.stato_app = "calcolo_13c"
            st.rerun()

elif st.session_state.stato_app in ["calcolo_1h", "calcolo_13c"]:
    
    if st.button("← Modifica Parametri Acquisizione", use_container_width=False):
        st.session_state.stato_app = "input"
        st.rerun()
        
    mol = Chem.MolFromSmiles(st.session_state.ultimo_smiles)
    if mol is None:
        st.error("Errore: Struttura non valida.")
    else:
        props = calcola_proprieta(mol)
        iupac, comune = ottieni_nomi_pubchem(st.session_state.ultimo_smiles)
        
        st.markdown("---")
        st.markdown("### Nomenclatura")
        st.markdown(f"**IUPAC**: {iupac}<br>**Comune**: {comune}", unsafe_allow_html=True)

        st.markdown("### Proprietà")
        c1, c2 = st.columns(2)
        c1.metric("Formula Bruta", props['formula'])
        c2.metric("Massa Molare (g/mol)", f"{props['mw']:.2f}")
        
        st.markdown("### Analisi Insaturazioni (DBE)")
        st.latex(rf"DBE = {props['formula_dbe_str']}")
        st.latex(rf"DBE = {props['formula_dbe_val_str']} = {props['dbe']:.1f}")
        st.caption("Il computo considera i nuclei tetravalenti (C, Si), trivalenti (N, P) e monovalenti (H, Alogeni). L'interferenza dovuta a nuclei con ipervalenza potenziale (S, P) richiede ispezione manuale.")
        
        st.markdown("### Ispezione Topologica")
        for commento in analizza_stereochimica(mol): st.markdown(f"- {commento}")
        for commento in analisi_accoppiamenti_avanzati(mol): st.markdown(f"- {commento}")

        p = st.session_state.parametri
        
        if st.session_state.stato_app == 'calcolo_1h':
            freq = p['freq']
            solv = p['solvente']
            plot_title = f'1H-NMR ({int(freq)} MHz, {solv})'
            x_range = [-0.5, 12.5]
            signals = stima_locale_1h(props['mol_h']) 
            nmr_type = '1h'
            tech = '1h'
        else:
            freq = p['freq']
            solv = p['solvente']
            tech = p['tech']
            plot_title = f'13C-NMR {tech} ({int(freq)} MHz, {solv})'
            x_range = [-10, 220]
            signals = stima_locale_13c(props['mol_no_h'])
            nmr_type = '13c'

        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:

            fig_mol_draw = plt.figure(dpi=300)
            ax_mol_draw = fig_mol_draw.add_subplot(111)
            for atom in mol.GetAtoms(): atom.SetProp('atomNote', str(atom.GetIdx() + 1))
            d2d = rdMolDraw2D.MolDraw2DCairo(1500, 1000)
            d2d.drawOptions().annotationFontScale = 0.9
            d2d.DrawMolecule(mol)
            d2d.FinishDrawing()
            ax_mol_draw.imshow(Image.open(io.BytesIO(d2d.GetDrawingText())))
            ax_mol_draw.axis('off')
            salva_pagina_uniforme(pdf, fig_mol_draw)

            if nmr_type == '1h':
                x_ppm = np.linspace(x_range[0], x_range[1], int(freq * 200))
                gamma_base = 0.0025 * (500.0 / freq)
            else:
                x_ppm = np.linspace(x_range[0], x_range[1], int(freq * 200))
                gamma_base = 0.5

            y_intensity = np.zeros_like(x_ppm)
            segnali_visibili = []

            for sig in signals:
                if nmr_type == '1h':
                    scambiato = (solv in ["D2O", "CD3OD"] and sig.get('is_exchangeable', False))
                    if scambiato: continue 
                    segnali_visibili.append(sig)
                    
                    delta = float(sig.get('delta', 1.0))
                    sub_peaks = genera_picchi(delta, sig.get('multiplicity', 's'), float(sig.get('integral', 1)), freq)
                    gamma_applicato = 0.06 if sig.get('is_exchangeable', False) else gamma_base
                    for p_shift, p_int in sub_peaks:
                        y_intensity += p_int / (1.0 + ((x_ppm - p_shift) / gamma_applicato)**2)
                        
                elif nmr_type == '13c':
                    n_h = sig.get('n_h', 0)
                    if tech == "DEPT-135":
                        if n_h == 0: continue
                        p_int = -1.0 if n_h == 2 else 1.0
                    elif tech == "DEPT-90":
                        if n_h != 1: continue
                        p_int = 1.0
                    elif tech == "APT":
                        p_int = 1.0 if n_h in [0, 2] else -1.0
                    else:
                        p_int = 1.0
                        
                    segnali_visibili.append(sig)
                    y_intensity += p_int / (1.0 + ((x_ppm - float(sig.get('delta', 1.0))) / gamma_base)**2)

            y_min = min(y_intensity) * 1.15 if min(y_intensity) < 0 else 0
            y_max = max(y_intensity) * 1.15 if np.any(y_intensity) else 1

            # Gestione della selezione riga per interattività
            # Verrà calcolata durante il rendering Streamlit, ma per il PDF usiamo lo spettro completo base
            fig_main = plt.figure(dpi=300)
            ax_spec = fig_main.add_subplot(111)
            ax_spec.plot(x_ppm, y_intensity, color=BORDEAUX, linewidth=1.5)
            if nmr_type == '13c' and tech in ["DEPT-135", "APT"]: ax_spec.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax_spec.set_xlim(x_range[1], x_range[0])
            ax_spec.set_ylim(y_min, y_max)
            ax_spec.set_xlabel('Chemical Shift (ppm)', fontsize=12)
            ax_spec.set_ylabel('Intensità', fontsize=12)
            ax_spec.set_title(plot_title, fontsize=14, fontweight='bold')
            ax_spec.spines['top'].set_visible(False)
            ax_spec.spines['right'].set_visible(False)
            salva_pagina_uniforme(pdf, fig_main)

            if nmr_type == '1h' and len(segnali_visibili) > 0:
                fig_zoom, axes = plt.subplots(1, len(segnali_visibili), dpi=300)
                if len(segnali_visibili) == 1: axes = [axes]
                signals_sorted = sorted(segnali_visibili, key=lambda x: float(x.get('delta', 0)), reverse=True)
                for i, (ax, sig) in enumerate(zip(axes, signals_sorted)):
                    delta = float(sig.get('delta', 1.0))
                    ax.plot(x_ppm, y_intensity, color=BORDEAUX, linewidth=2.0) 
                    width_zoom = 0.20 if sig.get('is_exchangeable', False) else (0.08 * (500.0 / freq))
                    ax.set_xlim(delta + width_zoom, delta - width_zoom)
                    mask = (x_ppm >= delta - width_zoom) & (x_ppm <= delta + width_zoom)
                    ax.set_ylim(0, (np.max(y_intensity[mask]) if np.any(mask) else 1) * 1.1)
                    ax.set_title(f"{delta:.2f} ppm\n{sig.get('multiplicity', 's')}, {int(float(sig.get('integral', 1)))}H", fontsize=10)
                    ax.get_yaxis().set_visible(False)
                    for spine in ['top', 'right', 'left']: ax.spines[spine].set_visible(False)
                salva_pagina_uniforme(pdf, fig_zoom)

            # Preparazione dati tabella
            df_data = []
            for sig in signals:
                scambiato = (nmr_type == '1h' and solv in ["D2O", "CD3OD"] and sig.get('is_exchangeable', False))
                scomparso_dept = False
                note_acc = sig['coupling_comment']
                
                if nmr_type == '13c':
                    n_h = sig.get('n_h', 0)
                    if tech == "DEPT-135" and n_h == 0: scomparso_dept = True; note_acc = "C quaternario (invisibile nel DEPT-135)"
                    if tech == "DEPT-90" and n_h != 1: scomparso_dept = True; note_acc = "Non CH (invisibile nel DEPT-90)"
                if scambiato: note_acc = f"Scambio H/D in {solv}"
                
                df_data.append({
                    'Shift (ppm)': "N/D" if scambiato or scomparso_dept else f"{sig['delta']:.2f}",
                    'Tipo': sig.get('tipo_c', 'H'),
                    'Molteplicità': sig['multiplicity'] if not (scambiato or scomparso_dept) else "-",
                    'Atomi': ", ".join(map(str, sig['atoms'])),
                    'Note': note_acc,
                    '_sort_val': float(sig['delta'])
                })
            
            df_signals_display = pd.DataFrame(df_data).sort_values(by='_sort_val', ascending=False).drop(columns=['_sort_val']).reset_index(drop=True)

        st.markdown("---")
        
        col_table, col_mol = st.columns([0.6, 0.4])
        
        with col_table:
            st.markdown("### Assegnazione Segnali (Clicca una riga)")
            event = st.dataframe(
                df_signals_display, 
                use_container_width=True, 
                selection_mode="single-row",
                on_select="rerun"
            )
        
        # Estrazione atomi e shift selezionati dalla tabella
        selected_atoms = []
        selected_delta = None
        if len(event.selection.rows) > 0:
            idx = event.selection.rows[0]
            row_data = df_signals_display.iloc[idx]
            atomi_str = row_data['Atomi']
            if atomi_str != "N/D":
                selected_atoms = [int(a) - 1 for a in atomi_str.split(", ")]
            try:
                selected_delta = float(row_data['Shift (ppm)'])
            except ValueError:
                selected_delta = None

        with col_mol:
            st.markdown("### Evidenziazione Strutturale")
            fig_highlight = plt.figure(dpi=300, figsize=(5, 5))
            ax_high = fig_highlight.add_subplot(111)
            
            for atom in mol.GetAtoms(): 
                atom.SetProp('atomNote', str(atom.GetIdx() + 1))
            
            d2d_high = rdMolDraw2D.MolDraw2DCairo(1500, 1500)
            d2d_high.drawOptions().annotationFontScale = 0.9
            
            # Colore bordeaux con opacità ridotta (alfa = 0.25)
            bordeaux_rgba = (107/255, 20/255, 34/255, 0.25)
            highlight_dict = {a: bordeaux_rgba for a in selected_atoms}
            
            d2d_high.DrawMolecule(mol, highlightAtoms=selected_atoms, highlightAtomColors=highlight_dict)
            d2d_high.FinishDrawing()
            
            ax_high.imshow(Image.open(io.BytesIO(d2d_high.GetDrawingText())))
            ax_high.axis('off')
            st.pyplot(fig_highlight)
            plt.close(fig_highlight)

        # Spettro Interattivo Plotly con aggiunta del riquadro sul multipletto se selezionato
        st.markdown("### Spettro Generato")
        fig_interattivo = go.Figure()
        fig_interattivo.add_trace(go.Scatter(x=x_ppm, y=y_intensity, mode='lines', line=dict(color=BORDEAUX, width=1.5)))
        if nmr_type == '13c' and tech in ["DEPT-135", "APT"]:
            fig_interattivo.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
        
        # Aggiunta del quadratino/riquadro sul multipletto in base alla selezione
        if selected_delta is not None and nmr_type == '1h':
            width_box = 0.15 * (500.0 / freq)
            fig_interattivo.add_vrect(
                x0=selected_delta + width_box, 
                x1=selected_delta - width_box,
                fillcolor=BORDEAUX, 
                opacity=0.18,
                layer="above", 
                line_width=1.5,
                line_color=BORDEAUX,
                annotation_text=f"Shift: {selected_delta:.2f} ppm",
                annotation_position="top left"
            )
        
        fig_interattivo.update_layout(
            title=plot_title, 
            xaxis_title="Chemical Shift (ppm)", 
            yaxis_title="Intensità", 
            xaxis=dict(autorange="reversed"), 
            plot_bgcolor='white', 
            hovermode='x', 
            height=600,
            font=dict(family="Palatino, serif")
        )
        fig_interattivo.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
        fig_interattivo.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0', showticklabels=False)
        st.plotly_chart(fig_interattivo, use_container_width=True)

        st.markdown("---")
        st.download_button("Download Report (PDF)", data=pdf_buffer.getvalue(), file_name="Report_NMR.pdf", mime="application/pdf", use_container_width=True)
