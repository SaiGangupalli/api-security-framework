import React, { useState } from 'react';
import APIInput from './components/APIInput';
import ResultsDisplay from './components/ResultsDisplay';

const App = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async (endpoint) => {
    try {
      // Clear previous results
      setResults(null);
      setError(null);
      setLoading(true);

      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ endpoint })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail?.message || 'Failed to analyze API');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    setError(null);
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl mb-4">API Security Analyzer</h1>
      <APIInput onSubmit={handleAnalyze} onReset={handleReset} error={error} />
      {loading && (
        <div className="mt-4 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-2 text-gray-600">Analyzing API...</p>
        </div>
      )}
      {results && <ResultsDisplay results={results} onReset={handleReset} />}
    </div>
  );
};

export default App;