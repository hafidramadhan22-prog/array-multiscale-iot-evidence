"""Small dependency-free Markdown tables for saved research summaries."""
import math


def markdown_table(frame):
    def cell(value):
        if isinstance(value,float):
            return f'{value:.6f}' if math.isfinite(value) else ''
        return str(value).replace('|',r'\|').replace('\n',' ')
    rows=['| '+' | '.join(cell(c) for c in frame.columns)+' |',
          '| '+' | '.join('---' for _ in frame.columns)+' |']
    rows+=['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)
