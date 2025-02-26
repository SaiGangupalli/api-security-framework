const path = require('path');

module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      // Completely remove the IgnorePlugin if it's causing issues
      webpackConfig.plugins = webpackConfig.plugins.filter(
        plugin => plugin.constructor.name !== 'IgnorePlugin'
      );

      // Resolve fallback for path
      webpackConfig.resolve.fallback = {
        ...webpackConfig.resolve.fallback,
        path: require.resolve("path-browserify")
      };

      // Add path aliases
      webpackConfig.resolve.alias = {
        ...webpackConfig.resolve.alias,
        '@': path.resolve(__dirname, 'src')
      };

      return webpackConfig;
    }
  },
  devServer: {
    setupMiddlewares: (middlewares, devServer) => {
      if (!devServer) {
        throw new Error('webpack-dev-server is not defined');
      }
      return middlewares;
    }
  }
};