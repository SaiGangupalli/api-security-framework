import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  LineChart, Line
} from 'recharts';

// Constants defined at the top level
const vulnTypes = [
  'sql_injection',
  'xss',
  'auth_bypass',
  'rate_limiting',
  'command_injection',
  'path_traversal'
];

const vulnDisplayNames = {
  'sql_injection': 'SQL Injection',
  'xss': 'XSS',
  'auth_bypass': 'Auth Bypass',
  'rate_limiting': 'Rate Limiting',
  'command_injection': 'CLI',
  'path_traversal': 'Path Traversal'
};

const JsonDisplay = ({ data }) => {
  const formatData = (value) => {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'string') return value;
    return JSON.stringify(value, null, 2);
  };

  return (
    <pre className="bg-gray-50 p-4 rounded-lg overflow-auto max-h-80 text-sm font-mono border border-gray-200">
      {formatData(data)}
    </pre>
  );
};

const ElasticsearchDetails = ({ es_details }) => {
  if (!es_details) return null;

  return (
    <div className="bg-blue-50 p-4 rounded-lg mb-4">
      <h4 className="font-medium text-blue-800 mb-2">Elasticsearch Query Details</h4>
      <div className="space-y-2">
        <div className="flex items-center">
          <span className="text-gray-600 w-24">Status:</span>
          <span className={`px-2 py-1 rounded-full text-sm font-medium ${
            es_details.status_code === 200 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
          }`}>
            {es_details.status_code || 'N/A'}
          </span>
        </div>
        <div className="flex items-center">
          <span className="text-gray-600 w-24">Query Time:</span>
          <span className="font-mono">{es_details.response_time?.toFixed(3) || 'N/A'}s</span>
        </div>
        {es_details.query_timestamp && (
          <div className="flex items-center">
            <span className="text-gray-600 w-24">Timestamp:</span>
            <span className="text-sm">{new Date(es_details.query_timestamp).toLocaleString()}</span>
          </div>
        )}
      </div>
    </div>
  );
};

const RiskScores = ({ risk_assessment }) => {
  const getRiskColor = (score) => {
    if (score > 0.7) return 'bg-red-600';
    if (score > 0.4) return 'bg-yellow-600';
    return 'bg-green-600';
  };

  const getRiskLevel = (score) => {
    if (score > 0.7) return 'Critical';
    if (score > 0.4) return 'Medium';
    return 'Low';
  };

  return (
    <div className="space-y-4 mb-6">
      {Object.entries(risk_assessment)
        .filter(([key]) => key.endsWith('_risk'))
        .map(([risk, score]) => (
          <div key={risk} className="flex items-center justify-between">
            <span className="text-gray-700 capitalize w-1/3">
              {risk.replace('_risk', '').replace('_', ' ')}
            </span>
            <div className="flex items-center gap-4 w-2/3">
              <div className="flex-grow h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full ${getRiskColor(score)} transition-all duration-500`}
                  style={{ width: `${score * 100}%` }}
                />
              </div>
              <div className="flex items-center gap-2 w-32">
                <span className="font-medium">{(score * 100).toFixed(1)}%</span>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  score > 0.7 ? 'bg-red-100 text-red-800' :
                  score > 0.4 ? 'bg-yellow-100 text-yellow-800' :
                  'bg-green-100 text-green-800'
                }`}>
                  {getRiskLevel(score)}
                </span>
              </div>
            </div>
          </div>
        ))}
    </div>
  );
};

const ConfidenceScores = ({ confidence_scores, overall_accuracy }) => (
  <div className="mb-6">
    <div className="flex justify-between items-center mb-4">
      <h4 className="font-medium text-gray-800">Analysis Confidence</h4>
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-600">Overall Accuracy:</span>
        <span className="font-medium text-blue-600">{(overall_accuracy * 100).toFixed(1)}%</span>
      </div>
    </div>
    <div className="grid grid-cols-2 gap-4">
      {Object.entries(confidence_scores).map(([component, score]) => (
        <div key={component} className="bg-gray-50 p-4 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-700 capitalize">
              {component.replace('_confidence', '').replace('_', ' ')}
            </span>
            <span className="font-medium">{(score * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 transition-all duration-500"
              style={{ width: `${score * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  </div>
);

const AnalysisCoverage = ({ analyzed_components }) => (
  <div>
    <h4 className="font-medium text-gray-800 mb-4">Analysis Coverage</h4>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {Object.entries(analyzed_components).map(([component, analyzed]) => (
        <div
          key={component}
          className={`p-4 rounded-lg flex items-center justify-between ${
            analyzed ? 'bg-green-50' : 'bg-red-50'
          }`}
        >
          <span className={`capitalize text-sm ${
            analyzed ? 'text-green-700' : 'text-red-700'
          }`}>
            {component.replace('_', ' ')}
          </span>
          <span className={`text-sm font-medium ${
            analyzed ? 'text-green-600' : 'text-red-600'
          }`}>
            {analyzed ? '✓' : '×'}
          </span>
        </div>
      ))}
    </div>
  </div>
);

const TestCase = ({ test }) => (
  <div className="mb-4 p-4 bg-gray-50 rounded-lg border border-gray-200 overflow-hidden">
    <div className="flex justify-between items-start gap-4">
      <h4 className="font-semibold text-lg text-blue-800 break-words flex-1">{test.name}</h4>
      <span className={`shrink-0 px-3 py-1 rounded-full text-sm font-medium ${
        test.priority === 'Critical' ? 'bg-red-100 text-red-800' :
        test.priority === 'High' ? 'bg-orange-100 text-orange-800' :
        'bg-yellow-100 text-yellow-800'
      }`}>
        {test.priority}
      </span>
    </div>

    <p className="text-gray-600 mt-2 break-words whitespace-pre-wrap">{test.description}</p>

    {test.steps && (
      <div className="mt-3">
        <h5 className="font-medium text-gray-700">Steps:</h5>
        <div className="bg-white rounded-lg p-4">
          <ul className="list-disc pl-5 space-y-2">
            {test.steps.map((step, idx) => (
              <li key={idx} className="text-sm text-gray-600 break-words">
                {step}
              </li>
            ))}
          </ul>
        </div>
      </div>
    )}

    {test.expected_results && (
      <div className="mt-3">
        <h5 className="font-medium text-gray-700">Expected Results:</h5>
        <div className="bg-white rounded-lg p-4">
          <p className="text-sm text-gray-600 break-words whitespace-pre-wrap">
            {test.expected_results}
          </p>
        </div>
      </div>
    )}

    {test.remediation && (
      <div className="mt-3">
        <h5 className="font-medium text-gray-700">Remediation:</h5>
        <div className="bg-white rounded-lg p-4">
          <p className="text-sm text-gray-600 break-words whitespace-pre-wrap">
            {test.remediation}
          </p>
        </div>
      </div>
    )}
  </div>
);

const TestSuite = ({ suite }) => (
  <div className="mb-6 p-6 bg-white rounded-lg shadow-sm hover:shadow transition-shadow overflow-hidden">
    <h3 className="text-xl font-bold text-gray-900 break-words">{suite.type}</h3>
    <p className="text-gray-600 mt-1 break-words whitespace-pre-wrap">{suite.description}</p>

    <div className="mt-4 space-y-4">
      {suite.test_cases?.map((test, idx) => (
        <TestCase key={idx} test={test} />
      ))}
    </div>
  </div>
);

const ModelPerformanceGraphs = () => {
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/model-performance');
        if (!response.ok) {
          throw new Error('Failed to fetch performance data');
        }
        const data = await response.json();
        if (data.status === 'success') {
          // Transform the data for visualization
          const transformedData = Object.entries(data.data).reduce((acc, [vuln, models]) => {
            acc[vuln] = {
              'Random Forest': models.random_forest || {},
              'Gradient Boosting': models.gradient_boosting || {},
              'Neural Network': models.neural_network || {},
              'SVM': models.svm || {}
            };
            return acc;
          }, {});
          setPerformanceData(transformedData);
        } else {
          throw new Error(data.message || 'Failed to get performance data');
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPerformanceData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 text-red-700 rounded-lg">
        Error loading performance data: {error}
      </div>
    );
  }

  if (!performanceData) return null;

  // Transform data for F1 score comparison
  const f1ScoreData = Object.entries(performanceData).map(([vuln, models]) => ({
    name: vulnDisplayNames[vuln] || vuln,
    'Random Forest': models['Random Forest'].f1,
    'Gradient Boosting': models['Gradient Boosting'].f1,
    'Neural Network': models['Neural Network'].f1,
    'SVM': models['SVM'].f1
  }));

  // Transform data for training time comparison
  const trainingTimeData = Object.entries(performanceData).map(([vuln, models]) => ({
    name: vulnDisplayNames[vuln] || vuln,
    'Random Forest': models['Random Forest'].training_time,
    'Gradient Boosting': models['Gradient Boosting'].training_time,
    'Neural Network': models['Neural Network'].training_time,
    'SVM': models['SVM'].training_time
  }));

  // Transform data for radar charts
  const getModelPerformanceData = (modelType) => {
    return Object.entries(performanceData).map(([vuln, models]) => ({
      vulnerability: vulnDisplayNames[vuln] || vuln,
      precision: models[modelType].precision * 100,
      recall: models[modelType].recall * 100,
      f1: models[modelType].f1 * 100
    }));
  };

  return (
    <div className="space-y-8 mt-6">
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">F1 Score Comparison</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={f1ScoreData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0.8, 1]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="Random Forest" fill="#8884d8" />
              <Bar dataKey="Gradient Boosting" fill="#82ca9d" />
              <Bar dataKey="Neural Network" fill="#ffc658" />
              <Bar dataKey="SVM" fill="#ff8042" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Training Time Comparison (seconds)</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={trainingTimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Random Forest" fill="#8884d8" />
              <Bar dataKey="Gradient Boosting" fill="#82ca9d" />
              <Bar dataKey="Neural Network" fill="#ffc658" />
              <Bar dataKey="SVM" fill="#ff8042" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

const ResultsDisplay = ({ results, onReset }) => {
  if (!results) return null;

  const { api_details = {}, es_details = null, risk_assessment = {}, test_cases = [] } = results;

  // Console log for debugging
  console.log('Risk Assessment Data:', risk_assessment);

  return (
    <div className="mt-8 max-w-7xl mx-auto">
      <div className="mb-4 flex justify-end">
        <button
          onClick={onReset}
          className="bg-gray-200 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-300 transition-colors"
        >
          Clear Results
        </button>
      </div>

      <div className="flex gap-6">
        {/* Left Column - API Details */}
        <div className="w-2/5">
          <div className="bg-white p-6 rounded-lg shadow sticky top-4">
            <h3 className="font-semibold mb-4 text-xl">API Details</h3>

            {/* Elasticsearch Query Details */}
            <ElasticsearchDetails es_details={es_details} />

            {/* API Details Section */}
            <div className="space-y-6">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium text-gray-800 mb-2">Request Information</h4>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <span className="text-gray-600 w-24">Method:</span>
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                      {api_details.request?.method || api_details.method || 'N/A'}
                    </span>
                  </div>
                  <div className="flex items-center">
                    <span className="text-gray-600 w-24">Endpoint:</span>
                    <span className="text-sm font-mono">
                      {api_details.request?.uri || 'N/A'}
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-gray-800 font-medium mb-2">Request Headers</h4>
                <JsonDisplay data={api_details.request?.headers || {}} />
              </div>

              {api_details.request?.body && (
                <div>
                  <h4 className="text-gray-800 font-medium mb-2">Request Body</h4>
                  <JsonDisplay data={api_details.request.body} />
                </div>
              )}

              <div>
                <h4 className="text-gray-800 font-medium mb-2">Response Headers</h4>
                <JsonDisplay data={api_details.response?.headers || {}} />
              </div>

              {api_details.response?.body && (
                <div>
                  <h4 className="text-gray-800 font-medium mb-2">Response Body</h4>
                  <JsonDisplay data={api_details.response.body} />
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column - Risk Assessment & Test Cases */}
        <div className="w-3/5">
          {/* Risk Assessment */}
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <h3 className="font-semibold text-xl text-gray-900 mb-6">Risk Assessment</h3>

            {/* Risk Scores */}
            <RiskScores risk_assessment={risk_assessment} />

            {/* Confidence Scores */}
            <ConfidenceScores
              confidence_scores={risk_assessment.confidence_scores || {}}
              overall_accuracy={risk_assessment.overall_accuracy || 0}
            />

            {/* Analysis Coverage */}
            <AnalysisCoverage analyzed_components={risk_assessment.analyzed_components || {}} />
          </div>

          {/* Model Performance Graphs */}
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <h3 className="font-semibold text-xl text-gray-900 mb-6">Model Performance Analysis</h3>
            <ModelPerformanceGraphs />
          </div>

          {/* Test Cases */}
          {test_cases && test_cases.length > 0 && (
            <div className="space-y-4">
              <h3 className="font-semibold text-lg">Security Test Cases</h3>
              {test_cases.map((suite, index) => (
                <TestSuite key={index} suite={suite} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;