import streamlit as st
from streamlit_ketcher import st_ketcher
import requests
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

st.set_page_config(page_title="Simulatore NMR", layout="wide")
st.title("Simulatore Spettri NMR Avanzato 🧪")
st.markdown("**Disegna la molecola, imposta i parametri e genera lo spettro con analisi stereochimica automatica.**")

# --- FUNZIONI CHIMICHE ---
def calcola_proprieta(mol):
    mol_h = Chem.AddHs(mol)
    c = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 6)
    h = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
    n = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 7)
    x = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53])
    dbe = c + 1 - (h / 2.0) + (n / 2.0) - (x / 2.0)
    mw = Descriptors.MolWt(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol_h)
    return {'formula': formula, 'mw': mw, 'dbe': dbe, 'n_h': h, 'mol_h': mol_h, 'mol_no_h': mol}

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
        return "Errore API", "Errore API"

def analizza_stereochimica(mol):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    commenti = []
    
    if chiral_centers:
        centri = ", ".join([str(idx + 1) for idx, _ in chiral_centers])
        commenti.append(f"⚠️ **Molecola Chirale**: Rilevati centri stereogenici agli atomi [{centri}].")
        
        ch2_diastereotopici = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 2:
                ch2_diastereotopici.append(str(atom.GetIdx() + 1))
        
        if ch2_diastereotopici:
            commenti.append(f"🔍 **Protoni Diastereotopici Potenziali**: I gruppi CH₂ sugli atomi [{', '.join(ch2_diastereotopici)}] risentono dell'intorno chirale. I loro due protoni sono diastereotopici, non isocroni, e tenderanno a mostrare chemical shift distinti e accoppiamento geminale (sistema AB o ABX), che il simulatore locale approssimerà come un unico multipletto.")
    else:
        commenti.append("✅ **Molecola Achirale**: Nessun centro stereogenico rilevato. I protoni dei gruppi CH₂ simmetrici sono **enantiotopici**, pertanto rimangono isocroni (hanno lo stesso chemical shift) nei comuni solventi achirali selezionati.")
        
    return commenti

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
            if rank not in groups:
                groups[rank] = []
            groups[rank].append(atom)

    signals = []
    shifts_visti = []

    for rank, h_atoms in groups.items():
        rep_h = h_atoms[0]
        integral = len(h_atoms)
        neighbor = rep_h.GetNeighbors()[0]
        c_indices = {h.GetNeighbors()[0].GetIdx() + 1 for h in h_atoms}

        if neighbor.GetIsAromatic(): shift = 7.3
        elif neighbor.GetAtomicNum() == 8: shift = 4.5
        elif neighbor.GetAtomicNum() == 7: shift = 2.5
        elif neighbor.GetAtomicNum() == 6:
            if neighbor.GetHybridization() == Chem.HybridizationType.SP2: shift = 5.5
            elif neighbor.GetHybridization() == Chem.HybridizationType.SP: shift = 2.8
            else:
                n_carbon_neighbors = sum(1 for a in neighbor.GetNeighbors() if a.GetAtomicNum() == 6)
                shift = 0.9 + (0.3 * n_carbon_neighbors)
        else: shift = 2.0

        while any(abs(shift - sv) < 0.05 for sv in shifts_visti):
            shift += 0.1
        shifts_visti.append(shift)

        vicini_h = 0
        if neighbor.GetAtomicNum() == 6:
            for c_neigh in neighbor.GetNeighbors():
                if c_neigh.GetAtomicNum() == 6:
                    for h_atom in c_neigh.GetNeighbors():
                        if h_atom.GetAtomicNum() == 1 and ranks[h_atom.GetIdx()] != rank:
                            vicini_h += 1

        mult_map = {0:'s', 1:'d', 2:'t', 3:'q', 4:'m', 5:'m', 6:'m', 7:'m', 8:'m', 9:'m'}
        mult = mult_map.get(vicini_h, 'm') if neighbor.GetAtomicNum() == 6 else 's'

        signals.append({'delta': shift, 'multiplicity': mult, 'integral': integral, 'atoms': list(c_indices), 'is_exchangeable': neighbor.GetAtomicNum() in [7, 8]})
    return signals

def stima_locale_13c(mol_no_h):
    ranks = list(Chem.CanonicalRankAtoms(mol_no_h, breakTies=False))
    groups = {}
    for atom in mol_no_h.GetAtoms():
        if atom.GetAtomicNum() == 6:
            rank = ranks[atom.GetIdx()]
            if rank not in groups:
                groups[rank] = []
            groups[rank].append(atom)

    signals = []
    shifts_visti = []

    for rank, c_atoms in groups.items():
        rep_c = c_atoms[0]
        integral = len(c_atoms)
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
        signals.append({'delta': shift, 'multiplicity': 's', 'integral': integral, 'atoms': [atom.GetIdx() + 1 for atom in c_atoms]})
    return signals

# --- EDITOR E UI ---
smiles = st_ketcher()

st.markdown("### ⚙️ Parametri di Acquisizione")
col_param1, col_param2 = st.columns(2)
with col_param1:
    freq_1h = st.selectbox("Risoluzione Spettrometro (MHz)", [300.0, 400.0, 500.0, 600.0, 800.0], index=2)
    freq_13c = freq_1h / 4
with col_param2:
    solvente = st.selectbox("Solvente Deuterato", ["CDCl3", "DMSO-d6", "D2O", "CD3OD"])

col1, col2 = st.columns(2)
with col1:
    btn_1h = st.button(f"Genera Spettro ¹H-NMR ({int(freq_1h)} MHz)", type="primary", use_container_width=True)
with col2:
    btn_13c = st.button(f"Genera Spettro ¹³C-NMR ({int(freq_13c)} MHz)", type="secondary", use_container_width=True)

if btn_1h or btn_13c:
    nmr_type = '1h' if btn_1h else '13c'
    
    if not smiles:
        st.warning("Disegna una molecola prima di procedere.")
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Errore struttura. Controlla il disegno in Ketcher.")
        else:
            props = calcola_proprieta(mol)
            iupac, comune = ottieni_nomi_pubchem(smiles)
            
            st.markdown("---")
            st.markdown(f"**Nome IUPAC:** {iupac} | **Nome Comune:** {comune}")
            
            commenti_stereo = analizza_stereochimica(mol)
            for commento in commenti_stereo:
                st.info(commento)

            signals = []
            if nmr_type == '1h':
                plot_title = f'Spettro ¹H-NMR a {int(freq_1h)} MHz in {solvente}'
                x_range = [-0.5, 12.5]
                plot_color = '#0077B6'
                mol_for_local_pred = props['mol_h']
                signals = stima_locale_1h(mol_for_local_pred) # Forza il calcolo locale per gestire i flag strutturali (is_exchangeable)
            elif nmr_type == '13c':
                plot_title = f'Spettro ¹³C-NMR a {int(freq_13c)} MHz in {solvente}'
                x_range = [-10, 220]
                plot_color = '#CC3311'
                mol_for_local_pred = props['mol_no_h']
                signals = stima_locale_13c(mol_for_local_pred)

            if not signals:
                st.error("Nessun segnale calcolato.")
            else:
                pdf_buffer = io.BytesIO()
                with PdfPages(pdf_buffer) as pdf:

                    # 1. Struttura 2D
                    fig_mol_draw = plt.figure(figsize=(8, 6))
                    ax_mol_draw = fig_mol_draw.add_subplot(111)
                    for atom in mol.GetAtoms():
                        atom.SetProp('atomNote', str(atom.GetIdx() + 1))
                    d2d = rdMolDraw2D.MolDraw2DCairo(int(fig_mol_draw.dpi * fig_mol_draw.get_figwidth()), int(fig_mol_draw.dpi * fig_mol_draw.get_figheight()))
                    d2d.drawOptions().annotationFontScale = 0.9
                    d2d.DrawMolecule(mol)
                    d2d.FinishDrawing()
                    img_2d = Image.open(io.BytesIO(d2d.GetDrawingText()))
                    ax_mol_draw.imshow(img_2d)
                    ax_mol_draw.axis('off')
                    ax_mol_draw.set_title("Struttura Molecolare", fontsize=14, fontweight='bold')
                    pdf.savefig(fig_mol_draw)
                    st.pyplot(fig_mol_draw)
                    plt.close(fig_mol_draw)

                    # 2. Spettro
                    if nmr_type == '1h':
                        x_ppm = np.linspace(x_range[0], x_range[1], int(freq_1h * 100))
                        gamma = 0.0025 * (500.0 / freq_1h)
                    else:
                        x_ppm = np.linspace(x_range[0], x_range[1], int(freq_13c * 160))
                        gamma = 0.5

                    y_intensity = np.zeros_like(x_ppm)
                    segnali_filtrati = []

                    for sig in signals:
                        if nmr_type == '1h' and solvente == "D2O" and sig.get('is_exchangeable', False):
                            continue # Scambio isotopico: il protone mobile scompare
                        
                        segnali_filtrati.append(sig)
                        delta = float(sig.get('delta', 1.0))
                        
                        if nmr_type == '1h':
                            sub_peaks = genera_picchi(delta, sig.get('multiplicity', 's'), float(sig.get('integral', 1)), freq_1h)
                        else:
                            sub_peaks = [(delta, float(sig.get('integral', 1)))]

                        for p_shift, p_int in sub_peaks:
                            y_intensity += p_int / (1.0 + ((x_ppm - p_shift) / gamma)**2)

                    fig_main = plt.figure(figsize=(15, 5))
                    ax_spec = fig_main.add_subplot(111)
                    ax_spec.plot(x_ppm, y_intensity, color=plot_color, linewidth=1.2)
                    ax_spec.set_xlim(x_range[1], x_range[0])
                    ax_spec.set_ylim(0, max(y_intensity) * 1.15 if np.any(y_intensity) else 1)
                    ax_spec.set_xlabel(r'$\delta$ (ppm)', fontsize=11, fontweight='bold')
                    ax_spec.set_ylabel('Intensità', fontsize=11, fontweight='bold')
                    ax_spec.set_title(plot_title, fontweight='bold')
                    ax_spec.grid(True, linestyle='--', alpha=0.4)
                    pdf.savefig(fig_main)
                    st.pyplot(fig_main)
                    plt.close(fig_main)

                    # 3. Tabella
                    if segnali_filtrati:
                        df_signals = pd.DataFrame(segnali_filtrati)
                        if 'is_exchangeable' in df_signals.columns:
                            df_signals = df_signals.drop(columns=['is_exchangeable'])
                        df_signals['atoms'] = df_signals['atoms'].apply(lambda x: ', '.join(map(str, x)))
                        df_signals.rename(columns={'delta': r'$\delta$ (ppm)', 'multiplicity': 'Molteplicità', 'integral': 'Integrale', 'atoms': 'Atomi'}, inplace=True)
                        df_signals = df_signals.sort_values(by=r'$\delta$ (ppm)', ascending=False).reset_index(drop=True)
                        st.table(df_signals)

                pdf_buffer.seek(0)
                st.download_button(label="📥 Scarica Report PDF", data=pdf_buffer, file_name="spettro_NMR_completo.pdf", mime="application/pdf", use_container_width=True)
