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

st.set_page_config(page_title="Simulatore NMR", layout="wide", initial_sidebar_state="expanded")

# --- COSTANTI ---
BORDEAUX = '#6B1422'

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
    
    formula_dbe_str = rf"{c} + 1 - \frac{{{h}}}{{2}} + \frac{{{n}}}{{2}} - \frac{{{x}}}{{2}}"
    
    return {'formula': formula, 'mw': mw, 'dbe': dbe, 'formula_dbe_str': formula_dbe_str, 'n_h': h, 'mol_h': mol_h, 'mol_no_h': mol}

def ottieni_nomi_pubchem(smiles):
    try:
        url_iupac = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/property/IUPACName/JSON"
        res_iupac = requests.get(url_iupac, timeout=5)
        iupac_name = res_iupac.json()['PropertyTable']['Properties'][0]['IUPACName'] if res_iupac.status_code == 200 else "Non disponibile"
        
        url_syn = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/synonyms/JSON"
        res_syn = requests.get(url_syn, timeout=5)
        common_name = res_syn.json()['InformationList']['Information'][0]['Synonym'][0] if res_syn.status_code == 200 else "Non disponibile"
        return iupac_name, common_name
    except Exception:
        return "Errore connessione", "Errore connessione"

def analizza_stereochimica(mol):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    commenti = []
    
    if chiral_centers:
        centri = ", ".join([str(idx + 1) for idx, _ in chiral_centers])
        commenti.append(f"Centri stereogenici rilevati agli atomi: {centri}.")
        
        ch2_diastereotopici = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 2:
                ch2_diastereotopici.append(str(atom.GetIdx() + 1))
        
        if ch2_diastereotopici:
            commenti.append(f"Protoni diastereotopici: I gruppi CH2 sugli atomi {', '.join(ch2_diastereotopici)} risentono dell'intorno chirale. I due protoni non sono isocroni e mostrano chemical shift distinti con accoppiamento geminale.")
    else:
        commenti.append("Molecola achirale: Nessun centro stereogenico. I protoni dei gruppi CH2 simmetrici sono enantiotopici e rimangono isocroni.")
        
    return commenti

def descrivi_accoppiamento(mult):
    mappa = {
        's': "Singoletto: Nessun accoppiamento vicinale.",
        'd': "Doppietto: Accoppiamento con 1 protone equivalente (J costante).",
        't': "Tripletto: Accoppiamento con 2 protoni equivalenti.",
        'q': "Quartetto: Accoppiamento con 3 protoni equivalenti.",
        'dd': "Doppio doppietto: Accoppiamento con 2 protoni non equivalenti (J1 != J2).",
        'm': "Multipletto: Sistema di spin complesso."
    }
    return mappa.get(mult.lower(), "Segnale non risolto.")

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

        signals.append({
            'delta': shift, 
            'multiplicity': mult, 
            'integral': integral, 
            'atoms': list(c_indices), 
            'is_exchangeable': neighbor.GetAtomicNum() in [7, 8],
            'coupling_comment': descrivi_accoppiamento(mult)
        })
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
        signals.append({
            'delta': shift, 
            'multiplicity': 's', 
            'integral': integral, 
            'atoms': [atom.GetIdx() + 1 for atom in c_atoms],
            'coupling_comment': descrivi_accoppiamento('s')
        })
    return signals

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("Parametri")
    freq_1h = st.selectbox("Frequenza 1H (MHz)", [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0], index=2)
    freq_13c = st.selectbox("Frequenza 13C (MHz)", [75.0, 100.0, 125.0, 150.0, 200.0, 250.0], index=2)
    solvente = st.selectbox("Solvente", ["CDCl3", "DMSO-d6", "D2O", "CD3OD"])
    
    st.markdown("---")
    btn_1h = st.button("Spettro 1H", type="primary", use_container_width=True)
    btn_13c = st.button("Spettro 13C", type="secondary", use_container_width=True)

# --- UI MAIN ---
st.title("Simulatore NMR")

smiles = st_ketcher()

if btn_1h or btn_13c:
    nmr_type = '1h' if btn_1h else '13c'
    
    if not smiles:
        st.markdown("**Attenzione:** Inserire una struttura molecolare valida.")
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.markdown("**Errore:** Struttura non valida.")
        else:
            props = calcola_proprieta(mol)
            iupac, comune = ottieni_nomi_pubchem(smiles)
            
            st.markdown("### Nomenclatura")
            st.markdown(f"- **IUPAC**: {iupac}")
            st.markdown(f"- **Comune**: {comune}")

            st.markdown("### Proprietà")
            c1, c2 = st.columns(2)
            c1.metric("Formula", props['formula'])
            c2.metric("Massa (g/mol)", f"{props['mw']:.2f}")
            
            st.markdown("**Insaturazioni (DBE)**")
            st.latex(rf"DBE = C + 1 - \frac{{H}}{{2}} + \frac{{N}}{{2}} - \frac{{X}}{{2}}")
            st.latex(rf"DBE = {props['formula_dbe_str']} = {props['dbe']:.1f}")
            
            st.markdown("### Stereochimica")
            commenti_stereo = analizza_stereochimica(mol)
            for commento in commenti_stereo:
                st.markdown(f"- {commento}")

            signals = []
            if nmr_type == '1h':
                plot_title = f'1H-NMR ({int(freq_1h)} MHz, {solvente})'
                x_range = [-0.5, 12.5]
                signals = stima_locale_1h(props['mol_h']) 
            elif nmr_type == '13c':
                plot_title = f'13C-NMR ({int(freq_13c)} MHz, {solvente})'
                x_range = [-10, 220]
                signals = stima_locale_13c(props['mol_no_h'])

            if not signals:
                st.markdown("**Errore:** Nessun segnale calcolato.")
            else:
                pdf_buffer = io.BytesIO()
                with PdfPages(pdf_buffer) as pdf:

                    # 1. Struttura 2D (Esportata nel PDF)
                    fig_mol_draw = plt.figure(figsize=(6, 4), dpi=300)
                    ax_mol_draw = fig_mol_draw.add_subplot(111)
                    for atom in mol.GetAtoms(): atom.SetProp('atomNote', str(atom.GetIdx() + 1))
                    d2d = rdMolDraw2D.MolDraw2DCairo(int(fig_mol_draw.dpi * fig_mol_draw.get_figwidth()), int(fig_mol_draw.dpi * fig_mol_draw.get_figheight()))
                    d2d.drawOptions().annotationFontScale = 0.9
                    d2d.DrawMolecule(mol)
                    d2d.FinishDrawing()
                    img_2d = Image.open(io.BytesIO(d2d.GetDrawingText()))
                    ax_mol_draw.imshow(img_2d)
                    ax_mol_draw.axis('off')
                    pdf.savefig(fig_mol_draw)
                    plt.close(fig_mol_draw)

                    # 2. Generazione Array di Dati
                    if nmr_type == '1h':
                        x_ppm = np.linspace(x_range[0], x_range[1], int(freq_1h * 200))
                        gamma = 0.0025 * (500.0 / freq_1h)
                    else:
                        x_ppm = np.linspace(x_range[0], x_range[1], int(freq_13c * 200))
                        gamma = 0.5

                    y_intensity = np.zeros_like(x_ppm)
                    segnali_filtrati = []

                    for sig in signals:
                        if nmr_type == '1h' and solvente == "D2O" and sig.get('is_exchangeable', False):
                            continue 
                        
                        segnali_filtrati.append(sig)
                        delta = float(sig.get('delta', 1.0))
                        
                        if nmr_type == '1h':
                            sub_peaks = genera_picchi(delta, sig.get('multiplicity', 's'), float(sig.get('integral', 1)), freq_1h)
                        else:
                            sub_peaks = [(delta, float(sig.get('integral', 1)))]

                        for p_shift, p_int in sub_peaks:
                            y_intensity += p_int / (1.0 + ((x_ppm - p_shift) / gamma)**2)

                    # 3. Spettro Interattivo UI (Plotly)
                    st.markdown("### Spettro")
                    fig_interattivo = go.Figure()
                    fig_interattivo.add_trace(go.Scatter(x=x_ppm, y=y_intensity, mode='lines', line=dict(color=BORDEAUX, width=1.5)))
                    fig_interattivo.update_layout(
                        title=plot_title,
                        xaxis_title="Chemical Shift (ppm)",
                        yaxis_title="Intensità",
                        xaxis=dict(autorange="reversed"),
                        plot_bgcolor='white',
                        hovermode='x',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    fig_interattivo.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
                    fig_interattivo.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0', showticklabels=False)
                    st.plotly_chart(fig_interattivo, use_container_width=True)

                    # 4. Spettro Statico (Solo per PDF)
                    fig_main = plt.figure(figsize=(15, 5), dpi=300)
                    ax_spec = fig_main.add_subplot(111)
                    ax_spec.plot(x_ppm, y_intensity, color=BORDEAUX, linewidth=1.5)
                    ax_spec.set_xlim(x_range[1], x_range[0])
                    ax_spec.set_ylim(0, max(y_intensity) * 1.15 if np.any(y_intensity) else 1)
                    ax_spec.set_xlabel('Chemical Shift (ppm)', fontsize=12)
                    ax_spec.set_ylabel('Intensità', fontsize=12)
                    ax_spec.set_title(plot_title, fontsize=14, fontweight='bold')
                    ax_spec.spines['top'].set_visible(False)
                    ax_spec.spines['right'].set_visible(False)
                    ax_spec.grid(True, linestyle=':', alpha=0.6)
                    pdf.savefig(fig_main)
                    plt.close(fig_main)

                    # 5. Dettaglio Multipletti ad alta risoluzione (Mostrati anche in UI)
                    n_peaks = len(segnali_filtrati)
                    if n_peaks > 0 and nmr_type == '1h':
                        st.markdown("### Multipletti")
                        fig_zoom, axes = plt.subplots(1, n_peaks, figsize=(max(3 * n_peaks, 6), 3.5), dpi=300)
                        if n_peaks == 1: axes = [axes]

                        signals_sorted = sorted(segnali_filtrati, key=lambda x: float(x.get('delta', 0)), reverse=True)

                        for i, (ax, sig) in enumerate(zip(axes, signals_sorted)):
                            delta = float(sig.get('delta', 1.0))
                            mult = sig.get('multiplicity', 's')
                            integ = int(float(sig.get('integral', 1)))
                            atoms = sig.get('atoms', [])
                            
                            ax.plot(x_ppm, y_intensity, color=BORDEAUX, linewidth=2.0) 
                            
                            width_zoom = 0.08 * (500.0 / freq_1h) 
                            ax.set_xlim(delta + width_zoom, delta - width_zoom)

                            mask = (x_ppm >= delta - width_zoom) & (x_ppm <= delta + width_zoom)
                            local_max = np.max(y_intensity[mask]) if np.any(mask) else 1
                            ax.set_ylim(0, local_max * 1.1)

                            ax.set_title(f"{delta:.2f} ppm\n{mult}, {integ}H\nAtomi: {', '.join(map(str, sorted(atoms)))}", fontsize=10)
                            ax.get_yaxis().set_visible(False)
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                            ax.grid(True, linestyle=':', alpha=0.5)
                            ax.set_xlabel("ppm", fontsize=9)

                        plt.tight_layout()
                        pdf.savefig(fig_zoom)
                        st.pyplot(fig_zoom)
                        plt.close(fig_zoom)

                    # 6. Tabella Assegnazione
                    st.markdown("### Assegnazione")
                    if segnali_filtrati:
                        df_signals = pd.DataFrame(segnali_filtrati)
                        if 'is_exchangeable' in df_signals.columns:
                            df_signals = df_signals.drop(columns=['is_exchangeable'])
                        df_signals['atoms'] = df_signals['atoms'].apply(lambda x: ', '.join(map(str, x)))
                        
                        df_signals.rename(columns={
                            'delta': 'Shift (ppm)', 
                            'multiplicity': 'Molteplicità', 
                            'integral': 'Integrale', 
                            'atoms': 'Atomi',
                            'coupling_comment': 'Commento Accoppiamento'
                        }, inplace=True)
                        
                        df_signals = df_signals.sort_values(by='Shift (ppm)', ascending=False).reset_index(drop=True)
                        st.dataframe(df_signals, use_container_width=True)

                st.markdown("---")
                pdf_buffer.seek(0)
                st.download_button(label="Download PDF", data=pdf_buffer, file_name=f"Report_NMR.pdf", mime="application/pdf", use_container_width=True)
