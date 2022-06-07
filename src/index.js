import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Route, Routes } from 'react-router-dom';

import App from './App';
import EngCalc from './pages/Engcalc';

const container = document.getElementById('root');
const root = createRoot(container);

root.render(
  (
    <BrowserRouter>
      <div>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/eng-calc" element={<EngCalc />} />
        </Routes>
      </div>
    </BrowserRouter>
  ),
);
