"""Aggregate and visualize evaluation results from the results folder."""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

STYLE_PATH = Path(__file__).resolve().parents[1] / 'config' / 'plotting' / 'plt.mplstyle'
plt.style.use(STYLE_PATH)

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'

METHOD_LABELS = {
    'bundle': 'Bundle',
    'cover_sheaf': 'Cover Sheaf',
    'diag_sheaf': 'Diag Sheaf',
    'flat_bundle': 'Flat Bundle',
    'federated': 'Federated',
    'personalized_federated': 'Pers. Federated',
    'optimal_transport': 'Optimal Transport',
    'vanilla': 'Vanilla',
}

EXCLUDE_METHODS = {'diag_sheaf'}

MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']


def load_results(results_dir: Path) -> pd.DataFrame:
    records = []
    for parquet_file in sorted(results_dir.rglob('eval_metrics.parquet')):
        method_key = parquet_file.parent.name
        if method_key not in METHOD_LABELS or method_key in EXCLUDE_METHODS:
            continue
        df = pd.read_parquet(parquet_file)
        row = df.iloc[0].to_dict()
        row['method'] = METHOD_LABELS[method_key]
        records.append(row)
    return pd.DataFrame(records)


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df[['method', 'FOSCTTM', 'KS_mean']].copy().set_index('method')
    table.columns = ['FOSCTTM', 'KS']
    return table


def build_long_df(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format DataFrame: one row per (method, K, agent) with CT and TW values.

    Falls back to mean columns when per-agent columns are absent.
    """
    agent_pat = re.compile(r'^(CT|TW)_K(\d+)_agent_(\d+)$')
    mean_pat = re.compile(r'^(CT|TW)_K(\d+)$')
    rows = []
    for _, row in df.iterrows():
        agent_vals: dict[tuple[int, int], dict[str, float]] = {}
        for col, val in row.items():
            m = agent_pat.match(str(col))
            if m and val is not None:
                metric, K, agent = m.group(1), int(m.group(2)), int(m.group(3))
                agent_vals.setdefault((K, agent), {})[metric] = float(val)

        if agent_vals:
            for (K, agent), metrics in agent_vals.items():
                rows.append({'method': row['method'], 'K': K, 'agent': agent, **metrics})
        else:
            mean_vals: dict[int, dict[str, float]] = {}
            for col, val in row.items():
                m = mean_pat.match(str(col))
                if m and val is not None:
                    metric, K = m.group(1), int(m.group(2))
                    mean_vals.setdefault(K, {})[metric] = float(val)
            for K, metrics in mean_vals.items():
                rows.append({'method': row['method'], 'K': K, 'agent': 0, **metrics})

    return pd.DataFrame(rows)


def _fmt(val, fmt='.3f') -> str:
    try:
        f = float(val)
        if pd.isna(f):
            return r'\text{--}'
        return format(f, fmt)
    except (TypeError, ValueError):
        return r'\text{--}'


def make_latex_table(df: pd.DataFrame) -> str:
    """NeurIPS/IEEE-style booktabs table with top-3 cell highlighting per metric.

    Requires in preamble:
        \\usepackage{booktabs}
        \\usepackage[table]{xcolor}
        \\definecolor{best}{RGB}{220,50,50}
        \\definecolor{second}{RGB}{255,165,80}
        \\definecolor{third}{RGB}{255,230,100}
    """
    # Metric config: (df_column, display_header, lower_is_better)
    metrics = [
        ('KS_mean', r'KS $\downarrow$', True),
        ('TW_K10',  r'TW$(K\!=\!10)$ $\uparrow$', False),
        ('FOSCTTM', r'FOSCTTM $\downarrow$', True),
    ]

    # Compute rank colours per metric: {(method, col): r'\cellcolor{...}'}
    cell_colour: dict[tuple[str, str], str] = {}
    rank_colours = ['best', 'second', 'third']
    for col, _, lower_is_better in metrics:
        series = df.set_index('method')[col].apply(
            lambda v: float(v) if v is not None and not pd.isna(float(v) if v is not None else float('nan')) else float('nan')
        )
        ranked = series.dropna().sort_values(ascending=lower_is_better)
        for rank, method in enumerate(ranked.index[:3]):
            cell_colour[(method, col)] = rf'\cellcolor{{{rank_colours[rank]}}}'

    import re as _re

    def _agent_std(row, pat: str) -> str:
        """Compute std over columns matching pat; return formatted string or empty."""
        vals = [
            float(v) for k, v in row.items()
            if _re.match(pat, str(k)) and v is not None and not pd.isna(float(v))
        ]
        if len(vals) < 2:
            return ''
        return _fmt(pd.Series(vals).std(), '.3f')

    def cell(method, col, row):
        colour = cell_colour.get((method, col), '')
        mean_str = _fmt(row.get(col))
        if mean_str == r'\text{--}':
            return rf'{colour}${mean_str}$'
        if col == 'KS_mean':
            std_str = _agent_std(row, r'^KS_agent_\d+$')
        elif col == 'TW_K10':
            std_str = _agent_std(row, r'^TW_K10_agent_\d+$')
        else:
            std_str = ''
        val = rf'{mean_str} \pm {std_str}' if std_str else mean_str
        return rf'{colour}${val}$'

    header_cols = ' & '.join(h for _, h, _ in metrics)
    lines = [
        r'% Preamble: \usepackage{booktabs}, \usepackage[table]{xcolor}',
        r'% \definecolor{best}{RGB}{220,50,50}',
        r'% \definecolor{second}{RGB}{255,165,80}',
        r'% \definecolor{third}{RGB}{255,230,100}',
        r'\begin{table}[ht]',
        r'  \centering',
        r'  \small',
        r'  \begin{tabular}{lrrr}',
        r'    \toprule',
        rf'    \textbf{{Method}} & {header_cols} \\',
        r'    \midrule',
    ]
    for _, row in df.iterrows():
        method = row['method']
        ks   = cell(method, 'KS_mean', row)
        tw10 = cell(method, 'TW_K10',  row)
        fosc = cell(method, 'FOSCTTM', row)
        lines.append(rf'    {method} & {ks} & {tw10} & {fosc} \\')
    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'  \caption{}',
        r'  \label{tab:results}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def plot_ct_metric(df: pd.DataFrame, out_path: Path):
    long = build_long_df(df)
    methods = df['method'].tolist()
    palette = dict(zip(methods, sns.color_palette('tab10', n_colors=len(methods))))
    markers = dict(zip(methods, MARKERS[: len(methods)]))

    fig, ax = plt.subplots(figsize=(9, 8))

    sns.lineplot(
        data=long,
        x='K',
        y='CT',
        hue='method',
        hue_order=methods,
        palette=palette,
        markers=markers,
        style='method',
        dashes=False,
        errorbar='sd',
        err_style='band',
        ax=ax,
    )

    ax.legend(
        framealpha=0.8,
        ncol=3,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        fontsize='small',
    )
    ax.set_xlabel(r'$K$')
    ax.set_ylabel(r'Continuity (CT)')
    ax.set_title(None)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    for suffix in ('.pdf', '.png'):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved CT plot → {path}')
    plt.close(fig)


def print_table(table: pd.DataFrame):
    print('\n=== Summary Table (FOSCTTM, KS) ===')
    display = table.copy()
    for col in display.columns:
        display[col] = display[col].apply(
            lambda x: f'{x:.4f}' if pd.notna(x) and x is not None else 'N/A'
        )
    print(display.to_string())
    print()


def save_table(table: pd.DataFrame, out_path: Path):
    table.to_csv(out_path.with_suffix('.csv'))
    print(f'Saved table CSV → {out_path.with_suffix(".csv")}')

    display = table.copy()
    for col in display.columns:
        display[col] = display[col].apply(
            lambda x: f'{float(x):.4f}' if pd.notna(x) and x is not None else 'N/A'
        )

    fig, ax = plt.subplots(figsize=(6, 1.2 + 0.5 * len(table)))
    ax.axis('off')
    tbl = ax.table(
        cellText=display.values,
        rowLabels=display.index,
        colLabels=display.columns,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.5, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#ecf0f1')

    fig.tight_layout()
    for suffix in ('.pdf', '.png'):
        path = out_path.with_suffix(suffix)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved table figure → {path}')
    plt.close(fig)


def main():
    df = load_results(RESULTS_DIR)
    print(f'Loaded {len(df)} methods: {df["method"].tolist()}')

    table = make_summary_table(df)
    print_table(table)

    out_dir = RESULTS_DIR
    save_table(table, out_dir / 'summary_table')

    latex = make_latex_table(df)
    tex_path = out_dir / 'results_table.tex'
    tex_path.write_text(latex)
    print(f'Saved LaTeX table → {tex_path}')
    print('\n' + latex + '\n')

    plot_ct_metric(df, out_dir / 'ct_plot')


if __name__ == '__main__':
    main()
