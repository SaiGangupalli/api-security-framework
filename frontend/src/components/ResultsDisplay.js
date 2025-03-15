import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  LineChart, Line
} from 'recharts';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/alert';
import { Play, CheckCircle, XCircle, Clock, Edit, Save, RotateCcw } from 'lucide-react';

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

const toNumber = (value, defaultValue = 0) => {
  const num = Number(value);
  return isNaN(num) ? defaultValue : num;
};

const RelevanceScale = ({ score }) => {
  const getColor = (score) => {
    if (score >= 8) return 'bg-blue-600';     // High relevance
    if (score >= 6) return 'bg-blue-500';     // Medium relevance
    if (score >= 4) return 'bg-blue-400';     // Low relevance
    return 'bg-blue-300';                     // Minimal relevance
  };

  const getLabel = (score) => {
    if (score >= 8.0) return 'High';
    if (score >= 6.0) return 'Medium';
    if (score >= 4.0) return 'Low';
    return 'Minimal';
  };

  return (
    <div className="flex items-center gap-2">
      <div className={`w-3 h-3 rounded-full ${getColor(score)}`} />
      <span className="text-sm font-medium text-gray-700">
        {getLabel(score)} Relevance ({score.toFixed(2)}/10.0)
      </span>
    </div>
  );
};

const ScoreBreakdown = ({ scores, className = "" }) => {
  if (!scores) return null;

  return (
    <div className={`p-3 bg-gray-50 rounded-lg text-sm ${className}`}>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="space-y-1">
          <div className="flex justify-between">
            <span className="text-gray-600">Risk-based Score:</span>
            <span className="font-medium">{scores.riskBased.toFixed(2)}/6.0</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500"
              style={{ width: `${(scores.riskBased / 6) * 100}%` }}
            />
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between">
            <span className="text-gray-600">Priority Score:</span>
            <span className="font-medium">{scores.priority.toFixed(2)}/1.0</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500"
              style={{ width: `${scores.priority * 100}%` }}
            />
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between">
            <span className="text-gray-600">Completeness Score:</span>
            <span className="font-medium">{scores.completeness.toFixed(2)}/3.0</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500"
              style={{ width: `${scores.completeness * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

const CompletenessIndicators = ({ details }) => {
  if (!details) return null;

  const indicators = [
    { label: 'Test Steps', status: details.hasSteps },
    { label: 'Expected Results', status: details.hasExpectedResults },
    { label: 'Remediation', status: details.hasRemediation }
  ];

  return (
    <div className="flex gap-4 mt-2">
      {indicators.map(({ label, status }) => (
        <div
          key={label}
          className={`px-2 py-1 rounded-full text-xs font-medium ${
            status
              ? 'bg-green-100 text-green-800'
              : 'bg-gray-100 text-gray-600'
          }`}
        >
          {status ? '✓' : '×'} {label}
        </div>
      ))}
    </div>
  );
};

const TestRelevanceDisplay = ({ test_cases, risk_assessment }) => {
  // Transform and sort test cases data for visualization
  const relevanceData = test_cases
    .flatMap(suite => {
      // Calculate suite-level scores consistently
      const assignedSuite = calculateSuiteScores(suite, risk_assessment);

      // Create data points for both suite and its test cases
      const suitePoint = {
        name: suite.type,
        fullName: suite.type, // Store the full name for tooltip
        relevance: parseFloat(assignedSuite.relevance_score?.toFixed(2) || "0"),
        risk: parseFloat((assignedSuite.risk_score * 100)?.toFixed(1) || "0")
      };

      const testPoints = assignedSuite.test_cases?.map(test => {
        // Store both shortened and full name
        const fullName = `${suite.type} - ${test.name}`;
        // Get name before hyphen or use full name if no hyphen
        const shortName = test.name.split(' - ')[0] || test.name;

        // FIX: Proper handling of test relevance scores
        let relevanceScore = 0;
        if (test.total !== undefined) {
          relevanceScore = test.total;
        } else if (test.relevance_score !== undefined) {
          relevanceScore = test.relevance_score;
        }

        // FIX: Proper handling of test risk scores
        let riskPercent = 0;
        if (test.breakdown?.riskBased) {
          riskPercent = (test.breakdown.riskBased / RISK_MULTIPLIER) * 100;
        } else if (test.risk_score !== undefined) {
          riskPercent = test.risk_score * 100;
        }

        return {
          name: shortName, // Use shortened name for display
          fullName: fullName, // Store full name for tooltip
          relevance: parseFloat(relevanceScore.toFixed(2) || "0"),
          risk: parseFloat(riskPercent.toFixed(1) || "0")
        };
      }) || [];

      return [suitePoint, ...testPoints];
    })
    .sort((a, b) => b.relevance - a.relevance);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      // Use the fullName for tooltip if available
      const displayName = payload[0].payload.fullName || label;
      return (
        <div className="bg-white p-3 rounded-lg shadow border">
          <p className="font-medium text-gray-900">{displayName}</p>
          <p className="text-blue-600">Relevance Score: {payload[0].value.toFixed(2)}</p>
          <p className="text-red-600">Risk Score: {payload[1].value.toFixed(1)}%</p>
        </div>
      );
    }
    return null;
  };

  // Calculate max relevance for scaling
  const maxRelevance = Math.max(...relevanceData.map(d => d.relevance), 3);

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
      <h3 className="text-xl font-bold text-gray-900 mb-4">Test Case Relevance Analysis</h3>
      <div className="h-96"> {/* Increased height to accommodate labels */}
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={relevanceData}
            margin={{ top: 20, right: 30, left: 20, bottom: 120 }} // Increased bottom margin
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="name"
              angle={-45}
              textAnchor="end"
              height={100} // Increased height for labels
              interval={0}
              tick={{
                fontSize: 12,
                fill: '#374151',
                dx: -8 // Adjust label position horizontally
              }}
            />
            <YAxis
              yAxisId="relevance"
              orientation="left"
              label={{
                value: 'Relevance Score',
                angle: -90,
                position: 'insideLeft',
                style: { textAnchor: 'middle' }
              }}
              domain={[0, maxRelevance]}
            />
            <YAxis
              yAxisId="risk"
              orientation="right"
              label={{
                value: 'Risk Score (%)',
                angle: 90,
                position: 'insideRight',
                style: { textAnchor: 'middle' }
              }}
              domain={[0, 100]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar
              yAxisId="relevance"
              dataKey="relevance"
              name="Relevance Score"
              fill="#3b82f6"
              radius={[4, 4, 0, 0]} // Rounded corners on top
            />
            <Bar
              yAxisId="risk"
              dataKey="risk"
              name="Risk Score (%)"
              fill="#ef4444"
              radius={[4, 4, 0, 0]} // Rounded corners on top
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6">
        <h4 className="font-semibold text-lg mb-3">Score Distribution</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {relevanceData.map((item) => (
            <div key={item.name} className="bg-gray-50 p-4 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium text-gray-700 break-words max-w-[60%]">
                  {item.fullName || item.name}
                </span>
                <div className="flex gap-4">
                  <span className="text-blue-600">{item.relevance.toFixed(2)}</span>
                  <span className="text-red-600">{item.risk.toFixed(1)}%</span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 transition-all duration-500"
                    style={{ width: `${(item.relevance / maxRelevance) * 100}%` }}
                  />
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-500 transition-all duration-500"
                    style={{ width: `${item.risk}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Constants for scoring
const RISK_MULTIPLIER = 6; // Maximum risk-based score

const PRIORITY_SCORES = {
  'Critical': 1.0,  // High impact vulnerabilities
  'High': 0.7,     // Significant vulnerabilities
  'Medium': 0.4,   // Moderate concerns
  'Low': 0.2       // Minor issues
};

const COMPLETENESS_SCORES = {
  steps: 1.2,           // Test steps more heavily weighted
  expectedResults: 0.9, // Expected results validation
  remediation: 0.9      // Remediation guidance
};

// Score ranges for labels and classifications
const SCORE_RANGES = {
  relevance: {
    High: 8.0,     // Critical + comprehensive tests
    Medium: 6.0,   // Important security tests
    Low: 4.0,      // Basic security checks
    Minimal: 0.0   // Limited value tests
  },
  risk: {
    Critical: 0.7, // High-risk vulnerabilities
    High: 0.4,    // Medium-risk issues
    Low: 0.0      // Low-risk concerns
  }
};

// Risk type mapping for better detection
const riskMapping = {
  'sql': 'sql_injection_risk',
  'injection': 'sql_injection_risk',
  'xss': 'xss_risk',
  'cross': 'xss_risk',
  'auth': 'auth_bypass_risk',
  'jwt': 'auth_bypass_risk',
  'rate': 'rate_limiting_risk',
  'limit': 'rate_limiting_risk',
  'command': 'command_injection_risk',
  'rce': 'command_injection_risk',
  'path': 'path_traversal_risk',
  'directory': 'path_traversal_risk'
};

const RelevanceScoreCalculator = () => {
  // Risk type mapping for better detection
  const riskMapping = {
    'sql': 'sql_injection_risk',
    'injection': 'sql_injection_risk',
    'xss': 'xss_risk',
    'cross': 'xss_risk',
    'auth': 'auth_bypass_risk',
    'jwt': 'auth_bypass_risk',
    'rate': 'rate_limiting_risk',
    'limit': 'rate_limiting_risk',
    'command': 'command_injection_risk',
    'rce': 'command_injection_risk',
    'path': 'path_traversal_risk',
    'directory': 'path_traversal_risk'
  };

  // Priority weights for risk scoring
  const priorityWeights = {
    'Critical': 1.5,
    'High': 1.25,
    'Medium': 1.0,
    'Low': 0.75
  };

  // Core scoring function used consistently throughout the application
  const calculateRelevanceScore = (test, riskAssessment = {}) => {
    // 1. Risk-based score (0-6 points)
    const riskScore = getRiskScore(test, riskAssessment);
    const riskBasedScore = Math.min(riskScore * 6, 6.0);

    // 2. Priority score (0-1 points)
    const priorityScore = PRIORITY_SCORES[test.priority] || PRIORITY_SCORES.Low;

    // 3. Completeness score (0-3 points)
    const completenessScore = calculateCompletenessScore(test);

    // Calculate total score and round to 2 decimal places
    const totalScore = parseFloat((riskBasedScore + priorityScore + completenessScore).toFixed(2));

    return {
      total: Math.min(totalScore, 10.0),
      breakdown: {
        riskBased: riskBasedScore,
        priority: priorityScore,
        completeness: completenessScore
      },
      components: {
        hasSteps: Array.isArray(test.steps) && test.steps.length > 0,
        hasExpectedResults: !!test.expected_results,
        hasRemediation: !!test.remediation
      }
    };
  };

  // In the getRiskScore function of your RelevanceScoreCalculator
  const getRiskScore = (test, riskAssessment) => {
    // Normalize test type and description for comparison
    const normalizedType = (test.type || '').toLowerCase();
    const normalizedDesc = (test.description || '').toLowerCase();
    const normalizedName = (test.name || '').toLowerCase();
    const combinedText = `${normalizedType} ${normalizedDesc} ${normalizedName}`;

    // Find matching risk type based on keywords with broader matching
    let matchedRisk = Object.entries(riskMapping).find(([key]) =>
      combinedText.includes(key)
    );

    // If no match found and this is likely an Auth Bypass test
    if (!matchedRisk &&
        (combinedText.includes('auth') ||
         combinedText.includes('bypass') ||
         combinedText.includes('authentication'))) {
      matchedRisk = ['auth', 'auth_bypass_risk'];
    }

    // If no match found and this is likely an SQLi test
    if (!matchedRisk &&
        (combinedText.includes('sql') ||
         combinedText.includes('injection') ||
         combinedText.includes('database'))) {
      matchedRisk = ['sql', 'sql_injection_risk'];
    }

    // If still no match, assign default based on priority
    if (!matchedRisk) {
      const priority = test.priority?.toLowerCase() || 'medium';
      let defaultRiskScore = 0.3; // Default medium risk

      if (priority === 'critical') defaultRiskScore = 0.8;
      else if (priority === 'high') defaultRiskScore = 0.6;
      else if (priority === 'low') defaultRiskScore = 0.2;

      return defaultRiskScore;
    }

    const riskType = matchedRisk ? matchedRisk[1] : null;
    const riskScore = riskType ? (riskAssessment[riskType] || 0.3) : 0.3; // Default to 0.3 if not found

    // Apply priority weighting
    const priorityWeight = priorityWeights[test.priority] || 1.0;
    return riskScore * priorityWeight;
  };

  const calculateCompletenessScore = (test) => {
    let score = 0;

    if (Array.isArray(test.steps) && test.steps.length > 0) {
      score += COMPLETENESS_SCORES.steps;
    }
    if (test.expected_results) {
      score += COMPLETENESS_SCORES.expectedResults;
    }
    if (test.remediation) {
      score += COMPLETENESS_SCORES.remediation;
    }

    return Math.min(score, 3.0);
  };

  // Calculate suite-level scores
  // Calculate suite-level scores with improved handling of priorities
  const calculateSuiteScores = (suite, riskAssessment) => {
    // Helper function to normalize and map risk levels to priority scores
    const getPriorityScore = (level) => {
      if (!level) return PRIORITY_SCORES['Medium']; // Default to Medium

      // Normalize by converting to lowercase and trimming
      const normalized = level.toString().toLowerCase().trim();

      if (normalized.includes('critical')) return PRIORITY_SCORES['Critical'];
      if (normalized.includes('high')) return PRIORITY_SCORES['High'];
      if (normalized.includes('medium')) return PRIORITY_SCORES['Medium'];
      if (normalized.includes('low')) return PRIORITY_SCORES['Low'];

      return PRIORITY_SCORES['Medium']; // Default if no match
    };

    // If the suite already has relevance_score and risk_score, use them
    if (suite.relevance_score !== undefined && suite.risk_score !== undefined) {
      // Just ensure the test cases have proper score properties
      const testScores = (suite.test_cases || []).map(test => {
        // If the test case already has scores, keep them
        if (test.total !== undefined && test.breakdown) {
          return test;
        }

        // Otherwise calculate scores or use fallbacks from test properties
        const total = test.relevance_score || 0;
        const riskBased = (test.risk_score || 0) * RISK_MULTIPLIER;

        // Properly handle priority using the helper function
        const priority = getPriorityScore(test.priority || test.risk_level);

        // Calculate completeness based on presence of key elements
        const hasSteps = Array.isArray(test.steps) && test.steps.length > 0;
        const hasExpectedResults = !!test.expected_results;
        const hasRemediation = !!test.remediation;
        const completeness =
          (hasSteps ? COMPLETENESS_SCORES.steps : 0) +
          (hasExpectedResults ? COMPLETENESS_SCORES.expectedResults : 0) +
          (hasRemediation ? COMPLETENESS_SCORES.remediation : 0);

        return {
          ...test,
          total: total,
          breakdown: {
            riskBased: riskBased,
            priority: priority,
            completeness: completeness
          },
          components: {
            hasSteps,
            hasExpectedResults,
            hasRemediation
          }
        };
      });

      return {
        ...suite,
        test_cases: testScores
      };
    }

    // Original implementation when scores need to be calculated from scratch
    const testScores = suite.test_cases?.map(test => ({
      ...test,
      ...calculateRelevanceScore(test, riskAssessment)
    })) || [];

    // Calculate aggregate suite scores
    const suiteScore = {
      total: Math.max(...testScores.map(t => t.total || 0), 0),
      riskScore: Math.max(...testScores.map(t => (t.breakdown?.riskBased || 0) / RISK_MULTIPLIER), 0),
      breakdown: {
        riskBased: Math.max(...testScores.map(t => t.breakdown?.riskBased || 0), 0),
        priority: Math.max(...testScores.map(t => t.breakdown?.priority || 0), 0),
        completeness: Math.max(...testScores.map(t => t.breakdown?.completeness || 0), 0)
      }
    };

    return {
      ...suite,
      relevance_score: suiteScore.total,
      risk_score: suiteScore.riskScore,
      score_breakdown: suiteScore.breakdown,
      test_cases: testScores
    };
  };

  return {
    calculateRelevanceScore,
    calculateSuiteScores,
    getRelevanceLabel: (score) => {
      if (score >= SCORE_RANGES.relevance.High) return 'High';
      if (score >= SCORE_RANGES.relevance.Medium) return 'Medium';
      if (score >= SCORE_RANGES.relevance.Low) return 'Low';
      return 'Minimal';
    },
    getRiskLabel: (score) => {
      if (score >= SCORE_RANGES.risk.Critical) return 'Critical';
      if (score >= SCORE_RANGES.risk.High) return 'High';
      return 'Low';
    }
  };
};

const {
  calculateRelevanceScore,
  calculateSuiteScores,
  getRelevanceLabel,
  getRiskLabel
} = RelevanceScoreCalculator();

const TestCase = ({ test, risk_assessment }) => {
  const [showDetails, setShowDetails] = useState(true);

  // Use the provided relevance score or calculate a new one
  const scoreDetails = test.total !== undefined ? test :
    calculateRelevanceScore(test, risk_assessment || {});

  const getRelevanceColor = (score) => {
    if (score >= 4) return 'bg-blue-600';
    if (score >= 3) return 'bg-blue-500';
    if (score >= 2) return 'bg-blue-400';
    return 'bg-blue-300';
  };

  const TestScoreBreakdown = () => (
      <div className="p-4 bg-gray-50 rounded-lg mt-4">
        <h6 className="font-medium text-gray-700 mb-3">Score Breakdown</h6>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Risk-based Score:</span>
              <span className="text-sm font-medium">
                {scoreDetails.breakdown.riskBased.toFixed(2)}/6.0
              </span>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-500"
                style={{ width: `${(scoreDetails.breakdown.riskBased / 6) * 100}%` }}
              />
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Priority Score:</span>
              <span className="text-sm font-medium">
                {scoreDetails.breakdown.priority.toFixed(2)}/1.0
              </span>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-500"
                style={{ width: `${scoreDetails.breakdown.priority * 100}%` }}
              />
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Completeness Score:</span>
              <span className="text-sm font-medium">
                {scoreDetails.breakdown.completeness.toFixed(2)}/3.0
              </span>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-500"
                style={{ width: `${(scoreDetails.breakdown.completeness / 3) * 100}%` }}
              />
            </div>
          </div>
        </div>
      </div>
  );

  return (
    <div className="mb-4 p-4 bg-white rounded-lg border border-gray-200 hover:shadow-sm transition-shadow">
      <div className="flex flex-col">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <h4 className="font-semibold text-lg text-blue-800 break-words">{test.name}</h4>

            <div className="flex flex-wrap gap-3 mt-2">
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                test.priority === 'Critical' ? 'bg-red-100 text-red-800' :
                test.priority === 'High' ? 'bg-orange-100 text-orange-800' :
                'bg-yellow-100 text-yellow-800'
              }`}>
                {test.priority}
              </span>

              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${getRelevanceColor(scoreDetails.total)}`} />
                <span className="text-sm text-gray-600">
                  {getRelevanceLabel(scoreDetails.total)} Relevance ({scoreDetails.total.toFixed(2)}/10.0)
                </span>
              </div>

              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  scoreDetails.breakdown.riskBased / RISK_MULTIPLIER > 0.7 ? 'bg-red-600' :
                  scoreDetails.breakdown.riskBased / RISK_MULTIPLIER > 0.4 ? 'bg-yellow-400' :
                  'bg-green-400'
                }`} />
                <span className="text-sm text-gray-600">
                  {((scoreDetails.breakdown.riskBased / RISK_MULTIPLIER) * 100).toFixed(1)}% risk
                </span>
              </div>
            </div>
          </div>

          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-blue-600 hover:text-blue-800 text-sm font-medium ml-4"
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>
        </div>

        <p className="text-gray-600 mt-3 break-words whitespace-pre-wrap">
          {test.description}
        </p>

        {showDetails && (
          <div className="mt-4 space-y-4">
            <TestScoreBreakdown />

            {test.steps && test.steps.length > 0 && (
              <div className="mt-4">
                <h5 className="font-medium text-gray-700 mb-2">Steps:</h5>
                <div className="bg-gray-50 rounded-lg p-4">
                  <ul className="list-disc pl-5 space-y-2">
                    {test.steps.map((step, idx) => (
                      <li key={idx} className="text-sm text-gray-600 break-words">{step}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {test.expected_results && (
              <div className="mt-4">
                <h5 className="font-medium text-gray-700 mb-2">Expected Results:</h5>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600 break-words whitespace-pre-wrap">
                    {test.expected_results}
                  </p>
                </div>
              </div>
            )}

            {test.remediation && (
              <div className="mt-4">
                <h5 className="font-medium text-gray-700 mb-2">Remediation:</h5>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600 break-words whitespace-pre-wrap">
                    {test.remediation}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const TestSuite = ({ suite, risk_assessment }) => {
  // Use the suite-level score calculation from RelevanceScoreCalculator
  const assignedSuite = calculateSuiteScores(suite, risk_assessment || {});

  // Extract scores from the suite level
  const relevanceScore = assignedSuite.relevance_score || 0;
  const riskScore = assignedSuite.risk_score || 0;
  const scoreBreakdown = assignedSuite.score_breakdown;
  const completenessDetails = {
    hasSteps: assignedSuite.test_cases.some(test => test.components?.hasSteps),
    hasExpectedResults: assignedSuite.test_cases.some(test => test.components?.hasExpectedResults),
    hasRemediation: assignedSuite.test_cases.some(test => test.components?.hasRemediation)
  };

  return (
    <div className="mb-6 p-6 bg-white rounded-lg shadow-sm hover:shadow transition-shadow">
      <div className="flex flex-col">
        <div className="mb-4">
          <div className="flex-1">
            <h3 className="text-xl font-bold text-gray-900">{suite.type}</h3>
            <div className="flex items-center gap-4 mt-2">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  relevanceScore >= 4 ? 'bg-blue-600' :
                  relevanceScore >= 3 ? 'bg-blue-500' :
                  relevanceScore >= 2 ? 'bg-blue-400' :
                  'bg-blue-300'
                }`} />
                <span className="text-sm text-gray-600">
                  {getRelevanceLabel(relevanceScore)} Relevance ({relevanceScore.toFixed(2)}/10.0)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  riskScore > 0.7 ? 'bg-red-600' :
                  riskScore > 0.4 ? 'bg-yellow-400' :
                  'bg-green-400'
                }`} />
                <span className="text-sm text-gray-600">
                  {(riskScore * 100).toFixed(1)}% risk
                </span>
              </div>
            </div>
          </div>
        </div>

        <p className="text-gray-600 mb-4">{suite.description}</p>

        <div className="space-y-4">
          {assignedSuite.test_cases?.map((test, idx) => (
            <TestCase
              key={idx}
              test={{
                ...test,
                // Pass all scores directly from the suite
                suite_relevance_score: assignedSuite.relevance_score,
                risk_score: assignedSuite.risk_score,
                score_breakdown: assignedSuite.score_breakdown,
                completeness_details: completenessDetails
              }}
              risk_assessment={risk_assessment}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

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

const TabPanel = ({ children, className = "" }) => (
  <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
    {children}
  </div>
);

const TabButton = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 font-medium rounded-lg transition-colors ${
      active
        ? 'bg-blue-600 text-white'
        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
    }`}
  >
    {children}
  </button>
);

const TestResult = ({ result }) => {
  const [expanded, setExpanded] = useState(false);

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'text-green-600';
      case 'failed':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-600" />;
      default:
        return <Clock className="w-5 h-5 text-gray-600" />;
    }
  };

  const resultStr = result.result ? JSON.stringify(result.result, null, 2) : "";
  const shouldTruncate = resultStr.length > 300;

  return (
    <div className="border rounded-lg p-4 mb-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {getStatusIcon(result.status)}
          <h4 className="font-medium text-gray-800">{result.test_name}</h4>
        </div>
        <span className={`${getStatusColor(result.status)} font-medium`}>
          {result.status}
        </span>
      </div>

      {result.error && (
        <Alert variant="destructive" className="mt-2">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{result.error}</AlertDescription>
        </Alert>
      )}

      {result.result && (
        <div className="mt-2 bg-gray-50 p-4 rounded-lg">
          <pre className="text-sm overflow-x-auto">
            {shouldTruncate && !expanded ? (
              <>
                {resultStr.substring(0, 300)}...
                <button
                  onClick={() => setExpanded(true)}
                  className="block mt-2 text-blue-600 hover:text-blue-800 text-xs"
                >
                  Show more
                </button>
              </>
            ) : (
              <>
                {resultStr}
                {shouldTruncate && (
                  <button
                    onClick={() => setExpanded(false)}
                    className="block mt-2 text-blue-600 hover:text-blue-800 text-xs"
                  >
                    Show less
                  </button>
                )}
              </>
            )}
          </pre>
        </div>
      )}
    </div>
  );
};

const CodeDisplay = ({ code, maxHeight = "h-96" }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Format the code by removing extra indentation
  const formattedCode = code.split('\n').map(line => line.trimRight()).join('\n');

  return (
    <div className="rounded-lg overflow-hidden">
      <div className="bg-gray-800 text-gray-100 p-4">
        <div className="flex justify-between items-center mb-2">
          <div className="flex space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-gray-400 hover:text-white text-sm"
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
        <pre className={`overflow-auto ${isExpanded ? 'h-full' : maxHeight} font-mono text-sm`}>
          <code>{formattedCode}</code>
        </pre>
      </div>
    </div>
  );
};

const TestScriptEditor = ({ script, onSave, onCancel }) => {
  const [editedScript, setEditedScript] = useState(script.script);

  return (
    <div className="bg-gray-900 text-gray-100 rounded-lg overflow-hidden mt-4">
      <div className="flex justify-between items-center p-2 bg-gray-800">
        <div className="flex space-x-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
        </div>
        <span className="text-xs text-gray-400">Edit Test Script</span>
        <div className="flex gap-2">
          <button
            onClick={() => onSave(editedScript)}
            className="text-gray-400 hover:text-white text-sm flex items-center gap-1"
          >
            <Save className="w-3 h-3" />
            Save
          </button>
          <button
            onClick={onCancel}
            className="text-gray-400 hover:text-white text-sm"
          >
            Cancel
          </button>
        </div>
      </div>
      <textarea
        value={editedScript}
        onChange={(e) => setEditedScript(e.target.value)}
        className="w-full p-4 bg-gray-900 text-gray-100 font-mono text-sm outline-none"
        style={{ height: '400px', resize: 'vertical' }}
      />
    </div>
  );
};

const TestScriptViewer = ({ script, relevanceScore = 0, onEdit }) => {
  const [expanded, setExpanded] = useState(false);

  const getPriorityStyle = (priority) => {
    switch (priority?.toLowerCase()) {
      case 'critical':
        return 'bg-red-100 text-red-800';
      case 'high':
        return 'bg-orange-100 text-orange-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-green-100 text-green-800';
    }
  };

  const getRelevanceStyle = (score) => {
    if (score >= 8.0) return 'bg-blue-100 text-blue-800';
    if (score >= 6.0) return 'bg-blue-100 text-blue-700';
    return 'bg-blue-50 text-blue-600';
  };

  return (
    <div className="bg-gray-50 rounded-lg p-4 mb-4 border border-gray-200">
      <div className="flex justify-between items-start mb-2">
        <div className="space-y-2">
          <h4 className="font-medium text-gray-800">{script.test_name}</h4>
          <div className="flex items-center gap-2">
            <span className={`px-2 py-1 text-xs rounded-full ${getPriorityStyle(script.priority)}`}>
              {script.priority || 'Low'}
            </span>
            <span className={`px-2 py-1 text-xs rounded-full ${getRelevanceStyle(relevanceScore)}`}>
              Relevance: {relevanceScore.toFixed(2)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onEdit(script)}
            className="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center gap-1"
          >
            <Edit className="w-4 h-4" />
            Edit
          </button>
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-blue-600 hover:text-blue-800 text-sm font-medium"
          >
            {expanded ? 'Hide Code' : 'Show Code'}
          </button>
        </div>
      </div>

      <p className="text-sm text-gray-600 mb-2 line-clamp-2">
        {script.description}
      </p>

      {expanded && (
        <div className="bg-gray-900 text-gray-100 rounded-lg overflow-hidden mt-4">
          <div className="flex justify-between items-center p-2 bg-gray-800">
            <div className="flex space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
            </div>
            <span className="text-xs text-gray-400">Test Script</span>
          </div>
          <pre className="p-4 overflow-x-auto font-mono text-sm whitespace-pre-wrap">
            <code>{script.script}</code>
          </pre>
        </div>
      )}
    </div>
  );
};

const TestAutomation = ({ test_cases = [], api_details, risk_assessment }) => {
  const [testScripts, setTestScripts] = useState([]);
  const [testResults, setTestResults] = useState([]);
  const [relevantCasesCount, setRelevantCasesCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [relevanceThreshold, setRelevanceThreshold] = useState(8.0);
  const [editingScript, setEditingScript] = useState(null);

  // Helper function to determine threshold label
  const getThresholdLabel = (threshold) => {
    if (threshold >= 8.0) return "high";
    if (threshold >= 6.0) return "medium";
    if (threshold >= 4.0) return "low";
    return "minimal";
  };

  // Helper function to get relevance label
  const getRelevanceLabel = (score) => {
    if (score >= SCORE_RANGES.relevance.High) return 'High';
    if (score >= SCORE_RANGES.relevance.Medium) return 'Medium';
    if (score >= SCORE_RANGES.relevance.Low) return 'Low';
    return 'Minimal';
  };

  // This function calculates risk score based on test properties and risk assessment
  const getRiskScore = (test, riskAssessment) => {
    // Normalize test type and description for comparison
    const normalizedType = (test.type || '').toLowerCase();
    const normalizedDesc = (test.description || '').toLowerCase();
    const normalizedName = (test.name || '').toLowerCase();
    const combinedText = `${normalizedType} ${normalizedDesc} ${normalizedName}`;

    // Find matching risk type based on keywords with broader matching
    let matchedRisk = Object.entries(riskMapping).find(([key]) =>
      combinedText.includes(key)
    );

    // If no match found and this is likely an Auth Bypass test
    if (!matchedRisk &&
        (combinedText.includes('auth') ||
         combinedText.includes('bypass') ||
         combinedText.includes('authentication'))) {
      matchedRisk = ['auth', 'auth_bypass_risk'];
    }

    // If no match found and this is likely an SQLi test
    if (!matchedRisk &&
        (combinedText.includes('sql') ||
         combinedText.includes('injection') ||
         combinedText.includes('database'))) {
      matchedRisk = ['sql', 'sql_injection_risk'];
    }

    // If still no match, assign default based on priority
    if (!matchedRisk) {
      const priority = test.priority?.toLowerCase() || 'medium';
      let defaultRiskScore = 0.3; // Default medium risk

      if (priority === 'critical') defaultRiskScore = 0.8;
      else if (priority === 'high') defaultRiskScore = 0.6;
      else if (priority === 'low') defaultRiskScore = 0.2;

      return defaultRiskScore;
    }

    const riskType = matchedRisk ? matchedRisk[1] : null;
    const riskScore = riskType ? (riskAssessment[riskType] || 0.3) : 0.3; // Default to 0.3 if not found

    // Apply priority weighting
    const priorityWeight = {
      'Critical': 1.5,
      'High': 1.25,
      'Medium': 1.0,
      'Low': 0.75
    }[test.priority] || 1.0;

    return riskScore * priorityWeight;
  };

  // Calculate completeness score based on test properties
  const calculateCompletenessScore = (test) => {
    let score = 0;

    if (Array.isArray(test.steps) && test.steps.length > 0) {
      score += COMPLETENESS_SCORES.steps;
    }
    if (test.expected_results) {
      score += COMPLETENESS_SCORES.expectedResults;
    }
    if (test.remediation) {
      score += COMPLETENESS_SCORES.remediation;
    }

    return Math.min(score, 3.0);
  };

  // Unified relevance score calculation
  const calculateRelevanceScore = (test, riskAssessment = {}) => {
    // 1. Risk-based score (0-6 points)
    const riskScore = getRiskScore(test, riskAssessment);
    const riskBasedScore = Math.min(riskScore * RISK_MULTIPLIER, 6.0);

    // 2. Priority score (0-1 points)
    const priorityScore = PRIORITY_SCORES[test.priority] || PRIORITY_SCORES.Low;

    // 3. Completeness score (0-3 points)
    const completenessScore = calculateCompletenessScore(test);

    // Calculate total score and round to 2 decimal places
    const totalScore = parseFloat((riskBasedScore + priorityScore + completenessScore).toFixed(2));

    return {
      total: Math.min(totalScore, 10.0),
      breakdown: {
        riskBased: riskBasedScore,
        priority: priorityScore,
        completeness: completenessScore
      },
      components: {
        hasSteps: Array.isArray(test.steps) && test.steps.length > 0,
        hasExpectedResults: !!test.expected_results,
        hasRemediation: !!test.remediation
      }
    };
  };

  // Get test cases filtered by relevance threshold and sorted by relevance
  const getRelevantTestCases = useCallback((cases, riskAssessment) => {
    // Flat list of all test cases with scores
    const allTestCases = cases.flatMap(suite => {
      // Get tests from the suite
      const testsFromSuite = suite.test_cases || [];

      return testsFromSuite.map(test => {
        // Check if test already has scores calculated
        if (test.total !== undefined) {
          return {
            ...test,
            type: suite.type,
            name: test.name || `${suite.type} Test`,
            description: test.description || suite.description,
            priority: test.priority || 'Medium',
            relevance_score: test.total,
            risk_score: (test.breakdown?.riskBased / RISK_MULTIPLIER) || 0
          };
        }

        // Calculate scores for tests without pre-calculated values
        const scores = calculateRelevanceScore(test, riskAssessment);
        return {
          ...test,
          type: suite.type,
          name: test.name || `${suite.type} Test`,
          description: test.description || suite.description,
          priority: test.priority || 'Medium',
          relevance_score: scores.total,
          risk_score: scores.breakdown.riskBased / RISK_MULTIPLIER,
          breakdown: scores.breakdown,
          components: scores.components
        };
      });
    });

    // Filter by threshold and sort by relevance
    const filteredCases = allTestCases
      .filter(test => test.relevance_score >= relevanceThreshold)
      .sort((a, b) => b.relevance_score - a.relevance_score);

    // Update the count
    setRelevantCasesCount(filteredCases.length);

    return {
      cases: filteredCases,
      count: filteredCases.length
    };
  }, [relevanceThreshold]);

  // Make sure test scripts can be executed
  const cleanScriptForExecution = (scriptContent) => {
    let cleanedScript = scriptContent;

    // Remove markdown and find Python imports
    const importIndex = scriptContent.indexOf('import ');
    if (importIndex > 0) {
      cleanedScript = scriptContent.substring(importIndex);
    }

    // Remove explanation sections
    const explanationIndex = cleanedScript.indexOf('### Explanation');
    if (explanationIndex > 0) {
      cleanedScript = cleanedScript.substring(0, explanationIndex).trim();
    }

    // Remove code block markers
    cleanedScript = cleanedScript.replace(/```python\n/g, '');
    cleanedScript = cleanedScript.replace(/```/g, '');

    // Add run_test wrapper if needed
    if (!cleanedScript.includes('async def run_test()')) {
      cleanedScript = `
import asyncio
import aiohttp
import json
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to run the main test
async def run_test():
    try:
        results = {}
        # Main test logic
${cleanedScript.split('\n').map(line => '        ' + line).join('\n')}

        return results
    except Exception as e:
        logger.error(f"Test execution error: {str(e)}")
        return {"error": str(e)}

# Allow importing this module without running the test
if __name__ == "__main__":
    asyncio.run(run_test())
`;
    }

    return cleanedScript;
  };

  // Generate test scripts for selected test cases
  const generateTestScripts = async () => {
    if (!test_cases?.length) return;

    setLoading(true);
    setError(null);

    try {
      // Get relevant test cases
      const { cases: relevantCases, count } = getRelevantTestCases(test_cases, risk_assessment);

      if (relevantCases.length === 0) {
        setTestScripts([]);
        setError('No test cases meet the minimum relevance threshold');
        return;
      }

      // Prepare test cases for API
      const enhancedTestCases = relevantCases.map(testCase => ({
        name: testCase.name,
        type: testCase.type,
        description: testCase.description,
        priority: testCase.priority,
        steps: testCase.steps || [],
        expected_results: testCase.expected_results || '',
        remediation: testCase.remediation || '',
        relevance_score: testCase.relevance_score,
        risk_score: testCase.risk_score,
        metadata: {
          relevance_score: testCase.relevance_score,
          risk_score: testCase.risk_score,
          priority: testCase.priority,
          type: testCase.type,
          breakdown: testCase.breakdown
        }
      }));

      // Limit to 3 test cases for performance
      const limitedTestCases = enhancedTestCases.slice(0, 3);

      // API request
      const endpoint = api_details?.request?.uri || '';
      const method = api_details?.method || api_details?.request?.method || 'POST';

      const apiDetailsObj = {
        method: method || "POST",
        request: {
          uri: endpoint,
          method: method || "POST",
          headers: api_details?.request?.headers || {},
          body: api_details?.request?.body || {}
        },
        response: api_details?.response || {}
      };

      // Mock API call - in real app, use fetch to endpoint
      // This is a simulation to avoid hitting the actual API
      console.log('Would send to API:', {
        endpoint: apiDetailsObj.request.uri,
        test_cases: limitedTestCases,
        api_details: apiDetailsObj
      });

      // Simulate API response with mock data
      const mockScripts = limitedTestCases.map(tc => ({
        test_name: tc.name,
        test_type: tc.type,
        description: tc.description,
        script: `
import asyncio
import aiohttp
import json
import logging

async def test_${tc.type.replace(/[^a-z0-9]/gi, '_').toLowerCase()}():
    """
    Test for ${tc.name}
    """
    url = "${endpoint}"

    # Headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # Test payload based on vulnerability type
    payload = {"test": "data"}

    async with aiohttp.ClientSession() as session:
        async with session.${method.toLowerCase()}(url, headers=headers, json=payload) as response:
            status = response.status
            response_text = await response.text()
            try:
                response_json = await response.json()
            except:
                response_json = {}

            # Analyze the response
            results = {
                "status_code": status,
                "vulnerable": status >= 400,
                "message": "Test completed successfully"
            }

            return results

async def run_test():
    return await test_${tc.type.replace(/[^a-z0-9]/gi, '_').toLowerCase()}()
`,
        priority: tc.priority
      }));

      // Save the generated scripts
      setTestScripts(mockScripts.map(script => ({
        ...script,
        relevance_score: limitedTestCases.find(tc => tc.name === script.test_name)?.relevance_score || 0,
        risk_score: limitedTestCases.find(tc => tc.name === script.test_name)?.risk_score || 0,
        original_script: script.script,
        cleaned_script: cleanScriptForExecution(script.script)
      })));

    } catch (err) {
      setError(err.message || 'Failed to generate test scripts');
    } finally {
      setLoading(false);
    }
  };

  // Handle script editing
  const handleEditScript = (script) => {
    setEditingScript(script);
  };

  const handleSaveScript = (updatedCode) => {
    setTestScripts(prev =>
      prev.map(script =>
        script.test_name === editingScript.test_name
          ? {
              ...script,
              script: updatedCode,
              cleaned_script: cleanScriptForExecution(updatedCode)
            }
          : script
      )
    );
    setEditingScript(null);
  };

  const handleCancelEdit = () => {
    setEditingScript(null);
  };

  // Execute test scripts
  const executeTests = async () => {
    setIsExecuting(true);
    setError(null);
    setTestResults([]);

    try {
      // Mock execution - in a real app this would be an API call
      setTimeout(() => {
        const mockResults = testScripts.map(script => ({
          test_name: script.test_name,
          test_type: script.test_type,
          status: Math.random() > 0.3 ? 'completed' : 'failed',
          result: {
            status_code: Math.floor(Math.random() * 2) === 0 ? 200 : 403,
            vulnerable: Math.random() > 0.5,
            details: "This is a mock test result"
          },
          error: Math.random() > 0.7 ? "Mock error message" : null
        }));

        setTestResults(mockResults);
        setIsExecuting(false);
      }, 2000);
    } catch (err) {
      setError(err.message || 'Failed to execute tests');
      setIsExecuting(false);
    }
  };

  // Reset scripts to original versions
  const resetScripts = () => {
    setTestScripts(prev =>
      prev.map(script => ({
        ...script,
        script: script.original_script,
        cleaned_script: cleanScriptForExecution(script.original_script)
      }))
    );
  };

  // Generate scripts when component mounts or criteria changes
  useEffect(() => {
    if (test_cases?.length > 0 && api_details?.request?.uri) {
      // Small delay to prevent immediate execution
      const timer = setTimeout(() => {
        generateTestScripts();
      }, 300);

      return () => clearTimeout(timer);
    }
  }, [test_cases, api_details?.request?.uri, relevanceThreshold]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h3 className="text-xl font-bold text-gray-900">API Test Automation</h3>
          <div className="flex items-center gap-4 mt-2">
            <p className="text-sm text-gray-600">
              {relevantCasesCount} {getThresholdLabel(relevanceThreshold)} relevance test cases selected
            </p>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600">Relevance Threshold:</label>
              <select
                value={relevanceThreshold}
                onChange={(e) => setRelevanceThreshold(Number(e.target.value))}
                className="text-sm border rounded-md px-2 py-1"
              >
                <option value={4.0}>Low (4.0+)</option>
                <option value={6.0}>Medium (6.0+)</option>
                <option value={8.0}>High (8.0+)</option>
              </select>
            </div>
          </div>
        </div>
        <div className="flex gap-2">
          {testScripts.length > 0 && (
            <button
              onClick={resetScripts}
              className="flex items-center gap-2 bg-gray-100 text-gray-800 px-4 py-2 rounded-lg hover:bg-gray-200"
            >
              <RotateCcw className="w-4 h-4" />
              Reset Scripts
            </button>
          )}
          <button
            onClick={executeTests}
            disabled={isExecuting || testScripts.length === 0}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isExecuting ? (
              <>
                <Clock className="w-4 h-4 animate-spin" />
                Executing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Tests ({testScripts.length})
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h4 className="font-medium text-gray-800">Test Scripts</h4>
          {editingScript ? (
            <TestScriptEditor
              script={editingScript}
              onSave={handleSaveScript}
              onCancel={handleCancelEdit}
            />
          ) : (
            testScripts.length > 0 ? (
              testScripts.map((script, index) => (
                <TestScriptViewer
                  key={index}
                  script={script}
                  relevanceScore={script.relevance_score}
                  onEdit={() => handleEditScript(script)}
                />
              ))
            ) : (
              <div className="text-gray-500 text-center py-8 bg-gray-50 rounded-lg">
                No test scripts available
              </div>
            )
          )}
        </div>

        <div className="space-y-4">
          <h4 className="font-medium text-gray-800">Test Results</h4>
          {testResults.length > 0 ? (
            testResults.map((result, index) => (
              <TestResult key={index} result={result} />
            ))
          ) : (
            <div className="text-gray-500 text-center py-8 bg-gray-50 rounded-lg">
              No test results available. Click "Run Tests" to start testing.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const CombinedTestCaseList = ({ test_cases, risk_assessment }) => {
  // First calculate scores for all suites
  const scoredSuites = test_cases.map(suite =>
    suite.relevance_score !== undefined ?
      suite :
      calculateSuiteScores(suite, risk_assessment || {})
  );

  // Create a flat list of all test cases with their suite information
  const allTestCases = scoredSuites.flatMap(suite => {
    // Get test cases from each suite
    return (suite.test_cases || []).map(test => ({
      test: {
        ...test,
        // Ensure we have the suite information
        suite_type: suite.type,
        suite_description: suite.description,
        // Pass scores directly
        suite_relevance_score: suite.relevance_score,
        risk_score: suite.risk_score,
      },
      suite: suite,
      // Get the relevance score (for sorting)
      relevance: test.total || 0,
      // Each test case needs a unique ID
      id: `${suite.type}-${test.name}-${Math.random().toString(36).substring(2, 9)}`
    }));
  });

  // Sort all test cases by relevance score in descending order
  const sortedTestCases = allTestCases.sort((a, b) => b.relevance - a.relevance);

  return (
    <div className="space-y-4">
      {sortedTestCases.map((item) => (
        <SingleTestCase key={item.id} item={item} />
      ))}
    </div>
  );
};

// Individual test case component with its own state
const SingleTestCase = ({ item }) => {
  const [showDetails, setShowDetails] = useState(false);

  const toggleDetails = () => {
    setShowDetails(!showDetails);
  };

  return (
    <div className="mb-4 p-4 bg-white rounded-lg border border-gray-200 hover:shadow-sm transition-shadow">
      <div className="flex flex-col">
        <div className="flex justify-between items-start">
          <div className="flex-1">
            {/* Show suite type as context */}
            <div className="text-sm text-blue-600 mb-1">{item.suite.type}</div>

            {/* Test name with relevance badge */}
            <h4 className="font-semibold text-lg text-blue-800 break-words">
              {item.test.name}
            </h4>

            <div className="flex flex-wrap gap-3 mt-2">
              {/* Priority badge */}
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                item.test.priority === 'Critical' ? 'bg-red-100 text-red-800' :
                item.test.priority === 'High' ? 'bg-orange-100 text-orange-800' :
                'bg-yellow-100 text-yellow-800'
              }`}>
                {item.test.priority}
              </span>

              {/* Relevance score */}
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  item.relevance >= 8 ? 'bg-blue-600' :
                  item.relevance >= 6 ? 'bg-blue-500' :
                  item.relevance >= 4 ? 'bg-blue-400' :
                  'bg-blue-300'
                }`} />
                <span className="text-sm text-gray-600">
                  {getRelevanceLabel(item.relevance)} Relevance ({item.relevance.toFixed(2)}/10.0)
                </span>
              </div>

              {/* Risk score */}
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  (item.test.breakdown?.riskBased / RISK_MULTIPLIER) > 0.7 ? 'bg-red-600' :
                  (item.test.breakdown?.riskBased / RISK_MULTIPLIER) > 0.4 ? 'bg-yellow-400' :
                  'bg-green-400'
                }`} />
                <span className="text-sm text-gray-600">
                  {((item.test.breakdown?.riskBased / RISK_MULTIPLIER || 0) * 100).toFixed(1)}% risk
                </span>
              </div>
            </div>
          </div>

          <button
            onClick={toggleDetails}
            className="text-blue-600 hover:text-blue-800 text-sm font-medium ml-4"
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>
        </div>

        {/* Test description */}
        <p className="text-gray-600 mt-3 break-words whitespace-pre-wrap">
          {item.test.description}
        </p>

        {/* Test details section */}
        {showDetails && (
          <div className="mt-4 space-y-4">
            {/* Score Breakdown */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h6 className="font-medium text-gray-700 mb-3">Score Breakdown</h6>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Risk-based Score:</span>
                    <span className="text-sm font-medium">
                      {item.test.breakdown?.riskBased.toFixed(2) || "0.00"}/6.0
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${((item.test.breakdown?.riskBased || 0) / 6) * 100}%` }}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Priority Score:</span>
                    <span className="text-sm font-medium">
                      {item.test.breakdown?.priority.toFixed(2) || "0.00"}/1.0
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${(item.test.breakdown?.priority || 0) * 100}%` }}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Completeness Score:</span>
                    <span className="text-sm font-medium">
                      {item.test.breakdown?.completeness.toFixed(2) || "0.00"}/3.0
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${((item.test.breakdown?.completeness || 0) / 3) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Steps */}
            {item.test.steps && item.test.steps.length > 0 && (
              <div>
                <h5 className="font-medium text-gray-700 mb-2">Steps:</h5>
                <div className="bg-gray-50 rounded-lg p-4">
                  <ul className="list-disc pl-5 space-y-2">
                    {item.test.steps.map((step, idx) => (
                      <li key={idx} className="text-sm text-gray-600 break-words">{step}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {/* Expected Results */}
            {item.test.expected_results && (
              <div>
                <h5 className="font-medium text-gray-700 mb-2">Expected Results:</h5>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600 break-words whitespace-pre-wrap">
                    {item.test.expected_results}
                  </p>
                </div>
              </div>
            )}

            {/* Remediation */}
            {item.test.remediation && (
              <div>
                <h5 className="font-medium text-gray-700 mb-2">Remediation:</h5>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600 break-words whitespace-pre-wrap">
                    {item.test.remediation}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const ResultsDisplay = ({ results, onReset }) => {
  const [activeTab, setActiveTab] = useState('risk');
  const [showApiDetails, setShowApiDetails] = useState(true);

  // First check if results exist
  if (!results) return null;

  // Then destructure the values from results
  const { api_details = {}, es_details = null, risk_assessment = {}, test_cases = [] } = results;

  useEffect(() => {
    console.log('Risk Assessment:', risk_assessment);
    console.log('Test Cases:', test_cases);
  }, [risk_assessment, test_cases]);

  const tabs = [
    { id: 'risk', label: 'Risk Assessment' },
    { id: 'performance', label: 'Model Performance' },
    { id: 'relevance', label: 'Test Case Relevance' },
    { id: 'tests', label: 'Security Tests' },
    { id: 'automation', label: 'Test Automation' }
  ];

  return (
    <div className="mt-8 max-w-7xl mx-auto">
      <div className="mb-4 flex justify-between items-center">
        <div className="flex gap-2">
          {tabs.map(tab => (
            <TabButton
              key={tab.id}
              active={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </TabButton>
          ))}
        </div>
        <button
          onClick={onReset}
          className="bg-gray-200 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-300 transition-colors"
        >
          Clear Results
        </button>
      </div>

      <div className="flex gap-6">
        {/* Left Column - API Details */}
        <div className="w-2/5" style={{ display: activeTab === 'automation' ? (showApiDetails ? 'block' : 'none') : 'block' }}>
          <div className="bg-white p-6 rounded-lg shadow sticky top-4">
            <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold mb-4 text-xl">API Details</h3>
                {activeTab === 'automation' && (
                        <button
                          onClick={() => setShowApiDetails(!showApiDetails)}
                          className="text-blue-600 hover:text-blue-800 text-sm"
                        >
                          {showApiDetails ? 'Hide Details' : 'Show Details'}
                        </button>
                      )}
            </div>
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

        {/* Right Column - Tabbed Content */}
        <div className={activeTab === 'automation' && !showApiDetails ? "w-full" : "w-3/5"}>
          {activeTab === 'risk' && (
            <TabPanel>
              <h3 className="font-semibold text-xl text-gray-900 mb-6">Risk Assessment</h3>
              <RiskScores risk_assessment={risk_assessment} />
              <ConfidenceScores
                confidence_scores={risk_assessment.confidence_scores || {}}
                overall_accuracy={risk_assessment.overall_accuracy || 0}
              />
              <AnalysisCoverage analyzed_components={risk_assessment.analyzed_components || {}} />
            </TabPanel>
          )}

          {activeTab === 'performance' && (
            <TabPanel>
              <h3 className="font-semibold text-xl text-gray-900 mb-6">Model Performance Analysis</h3>
              <ModelPerformanceGraphs />
            </TabPanel>
          )}

          {activeTab === 'relevance' && test_cases.length > 0 && (
            <TabPanel className="mb-0">
              <TestRelevanceDisplay test_cases={test_cases} risk_assessment={risk_assessment} />
            </TabPanel>
          )}

          {activeTab === 'tests' && test_cases.length > 0 && (
            <TabPanel>
              <h3 className="font-semibold text-lg mb-4">Security Test Cases</h3>
              <CombinedTestCaseList test_cases={test_cases} risk_assessment={risk_assessment} />
            </TabPanel>
          )}

          {activeTab === 'automation' && (
            <TabPanel>
              <h3 className="font-semibold text-lg mb-4">Test Automation</h3>
              {test_cases.length > 0 ? (
                <TestAutomation
                  test_cases={test_cases}
                  api_details={api_details}
                  risk_assessment={risk_assessment}
                />
              ) : (
                <div className="text-gray-500 text-center py-8">
                  No test cases available for automation.
                </div>
              )}
            </TabPanel>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;