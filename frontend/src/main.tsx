import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@cloudscape-design/global-styles/index.css';
import { applyBrandTheme } from './lib/brandTheme';
import App from './App';

// Apply the brand theme before first render so components mount pre-themed.
applyBrandTheme();

const queryClient = new QueryClient();
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
