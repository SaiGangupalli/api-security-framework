import React, { useState } from 'react';

const APIInput = ({ onSubmit, onReset, error }) => {
  const [endpoint, setEndpoint] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(endpoint);
  };

  const handleReset = () => {
    setEndpoint('');
    onReset();
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="bg-white shadow-lg rounded-lg p-6">
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-gray-700 font-medium mb-2">
              API Endpoint URL
            </label>
            <input
              type="text"
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500"
              placeholder="https://api.example.com/endpoint"
              required
            />
          </div>
          {error && (
            <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-lg">
              {error}
            </div>
          )}
          <div className="flex gap-4">
            <button
              type="submit"
              className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Analyze
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="flex-1 bg-gray-200 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-300 transition-colors"
            >
              Reset
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default APIInput;