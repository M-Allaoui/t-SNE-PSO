import logo from './logo.svg';
import './App.css';
import React from "react";
import TSNEPSOVisualization from "./components/TSNEPSOVisualization";

function App() {
    return (
        <div>
            <h1>t-SNE-PSO: Interactive Embedding Visualization</h1>
            <TSNEPSOVisualization />
        </div>
    );
}

export default App;
