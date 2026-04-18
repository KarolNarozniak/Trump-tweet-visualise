// @ts-check

const config = {
  title: "Trump Graph",
  tagline: "Stable, time-dependent mention-network analytics",

  url: "http://localhost:3002",
  baseUrl: "/",
  trailingSlash: false,

  organizationName: "local",
  projectName: "trump-graph",
  onBrokenLinks: "throw",
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn"
    }
  },

  i18n: {
    defaultLocale: "en",
    locales: ["en"]
  },

  presets: [
    [
      "classic",
      {
        docs: {
          routeBasePath: "/",
          sidebarPath: require.resolve("./sidebars.js")
        },
        blog: false,
        pages: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css")
        }
      }
    ]
  ],
  plugins: [
    function disableBrokenWebpackBar() {
      return {
        name: "disable-broken-webpackbar",
        configureWebpack(config, _isServer, utils) {
          if (!config.plugins || utils.currentBundler.name !== "webpack") {
            return {};
          }

          const filteredPlugins = config.plugins.filter(
            (plugin) => plugin?.constructor?.name !== "WebpackBarPlugin"
          );
          return {
            mergeStrategy: { plugins: "replace" },
            plugins: filteredPlugins
          };
        }
      };
    }
  ],

  themeConfig: {
    navbar: {
      title: "Trump Graph Docs",
      items: [
        {
          type: "docSidebar",
          sidebarId: "mainSidebar",
          position: "left",
          label: "Documentation"
        }
      ]
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Project",
          items: [
            {
              label: "README",
              to: "/"
            }
          ]
        }
      ],
      copyright: `Copyright (c) ${new Date().getFullYear()} Trump Graph`
    }
  }
};

module.exports = config;
