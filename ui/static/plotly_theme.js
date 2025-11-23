window.nanoPlotlyTheme = {
    toLayoutOverrides(baseLayout = {}) {
        return {
            ...baseLayout,
            margin: { l: 60, r: 24, t: 60, b: 60, ...(baseLayout.margin || {}) },
            template: (baseLayout.template || 'plotly_white'),
            paper_bgcolor: 'rgba(255,255,255,0)',
            plot_bgcolor: 'rgba(248,250,252,0.75)',
            font: {
                family: 'Noto Sans SC, Helvetica, Arial, sans-serif',
                ...(baseLayout.font || {}),
            },
        };
    },

    toConfigOverrides(baseConfig = {}) {
        return {
            displaylogo: false,
            responsive: true,
            ...baseConfig,
        };
    },
};
