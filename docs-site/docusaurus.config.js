// @ts-check

const config = {
  title: "Trump Tweet Visualize",
  tagline: "Stable, time-dependent mention-network analytics",

  url: "http://localhost:3002",
  baseUrl: "/",
  trailingSlash: false,

  organizationName: "local",
  projectName: "trump-tweet-visualise",
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

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

  themeConfig: {
    navbar: {
      title: "Trump Tweet Visualize Docs",
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
      copyright: `Copyright © ${new Date().getFullYear()} Trump Tweet Visualize`
    }
  }
};

module.exports = config;
