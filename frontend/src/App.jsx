import { useState } from 'react'
import './App.css'

const API_URL = '/api/analyze'

function FindingsSection({ title, items, type }) {
return ( <section className="findings-section"> <div className="section-heading"> <h3>{title}</h3> <span className="count">{items.length}</span> </div>


  {items.length === 0 ? (
    <p className="empty-result">No findings in this category.</p>
  ) : (
    <div className="finding-list">
      {items.map((item, index) => (
        <article
          className="finding-card"
          key={`${item.rule_id || type}-${index}`}
        >
          <div className="finding-top">
            <span
              className={`severity severity-${(item.severity || 'info').toLowerCase()}`}
            >
              {item.severity || 'info'}
            </span>
            <span className="finding-location">
              Line {item.line ?? '—'} · Column {item.column ?? '—'}
            </span>
          </div>

          <h4>{item.message || item.title || 'Finding'}</h4>

          {item.rule_id && (
            <p className="rule-id">{item.rule_id}</p>
          )}

          {item.confidence != null && (
            <p className="confidence">
              Confidence: {(item.confidence * 100).toFixed(1)}%
            </p>
          )}

          {item.evidence && (
            <pre className="evidence">
              <code>{item.evidence}</code>
            </pre>
          )}

          {item.suggestion && (
            <p className="suggestion">{item.suggestion}</p>
          )}
        </article>
      ))}
    </div>
  )}
</section>


)
}

export default function App() {
const [code, setCode] = useState(
'def calculate_average(numbers):\n    total = 0\n    for number in numbers:\n        total += number\n    return total / len(numbers)\n'
)
const [report, setReport] = useState(null)
const [error, setError] = useState('')
const [loading, setLoading] = useState(false)

async function analyzeCode() {
if (!code.trim()) {
setError('Please enter some Python code to analyze.')
setReport(null)
return
}


setLoading(true)
setError('')
setReport(null)

try {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code })
  })

  const data = await response.json()

  if (!response.ok || data.status === 'error') {
    const message =
      typeof data.error === 'string'
        ? data.error
        : data.error?.message || JSON.stringify(data.error)

    setError(message || 'Analysis failed. Please check your code.')
    return
  }

  setReport(data)
} catch {
  setError(
    'Could not connect to the Python backend. Make sure api.py is running on port 5000.'
  )
} finally {
  setLoading(false)
}


}

const summary = report?.summary || {}

return ( <main className="app-shell"> <header className="topbar"> <a className="brand" href="#"> <span className="brand-icon">&lt;/&gt;</span> <span>
Code<span className="brand-accent">Lens</span> </span> </a>


    <span className="topbar-label">AI-POWERED STATIC ANALYSIS</span>

    <span className="status-indicator">
      <span />
      Python Analyzer
    </span>
  </header>

  <section className="hero">
    <p className="eyebrow">INTELLIGENT CODE REVIEW</p>
    <h1>
      Understand your code.
      <br />
      <span>Improve every line.</span>
    </h1>
    <p className="hero-description">
      Analyze Python code for potential bugs, complexity issues, and
      optimization opportunities.
    </p>
  </section>

  <section className="workspace">
    <div className="editor-panel">
      <div className="panel-header">
        <div className="panel-title">
          <span className="python-icon">Py</span>
          <div>
            <h2>Code Editor</h2>
            <p>Paste or edit your Python code</p>
          </div>
        </div>
        <span className="file-label">main.py</span>
      </div>

      <div className="editor-body">
        <div className="line-gutter" aria-hidden="true">
          {code.split('\n').map((_, index) => (
            <span key={index}>{index + 1}</span>
          ))}
        </div>

        <textarea
          aria-label="Python source code"
          spellCheck="false"
          value={code}
          onChange={(event) => setCode(event.target.value)}
          placeholder="Paste your Python code here..."
        />
      </div>

      <div className="editor-footer">
        <span>Python 3 · Static analysis</span>
        <span>{code.split('\n').length} lines</span>
      </div>

      <button
        className="analyze-button"
        onClick={analyzeCode}
        disabled={loading}
      >
        {loading ? 'Analyzing code...' : '▶ Analyze Code'}
      </button>

      <p></p>
    </div>

    <div className="results-panel">
      <div className="results-header">
        <div>
          <p className="eyebrow">ANALYSIS REPORT</p>
          <h2>Findings</h2>
        </div>
        {report && (
          <span className="complete-label">Analysis complete</span>
        )}
      </div>

      {error && (
        <div className="error-box" role="alert">
          <strong>Analysis could not be completed</strong>
          <p>{error}</p>
        </div>
      )}

      {!report && !error && (
        <div className="results-placeholder">
          <div className="placeholder-icon">{'{ }'}</div>
          <h3>Ready to analyze</h3>
          <p>
            Run an analysis to see bug detection, complexity estimates,
            and optimization suggestions here.
          </p>
        </div>
      )}

      {report && (
        <>
          <div className="summary-grid">
            <div className="summary-card">
              <span>Total findings</span>
              <strong>{summary.total_findings ?? 0}</strong>
            </div>

            <div className="summary-card">
              <span>Potential bugs</span>
              <strong>
                {summary.bugs ?? report.bugs?.length ?? 0}
              </strong>
            </div>

            <div className="summary-card">
              <span>Complexity</span>
              <strong>
                {summary.complexity_findings ??
                  report.complexity?.length ??
                  0}
              </strong>
            </div>

            <div className="summary-card">
              <span>Optimizations</span>
              <strong>
                {summary.optimization_recommendations ??
                  report.optimizations?.length ??
                  0}
              </strong>
            </div>
          </div>

          <FindingsSection
            title="Potential Bugs"
            items={report.bugs || []}
            type="bug"
          />

          <FindingsSection
            title="Complexity Analysis"
            items={report.complexity || []}
            type="complexity"
          />

          <FindingsSection
            title="Optimization Suggestions"
            items={report.optimizations || []}
            type="optimization"
          />
        </>
      )}
    </div>
  </section>

  <footer className="app-footer">
    <span>CodeLens · AI Code Analyzer</span>
    <span>Built with React + Python</span>
  </footer>
</main>


)
}
