import React, { useState, useEffect } from "react";
import Plot from "react-plotly.js";

const TSNEPSOVisualization = () => {
    const [data, setData] = useState([]);

    useEffect(() => {
        fetch("/tsne-pso-results.json")
            .then((response) => response.json())
            .then((result) => setData(result));
    }, []);

    return (
        <div>
            <h2>t-SNE-PSO Interactive Visualization</h2>
            <Plot 
                data={[{ x: data.x, y: data.y, type: "scatter", mode: "markers", marker: { color: data.labels } }]} 
                layout={{ title: "t-SNE-PSO Projection" }} 
            />
        </div>
    );
};

export default TSNEPSOVisualization;
