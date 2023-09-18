import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import {QueryClient, QueryClientProvider} from "react-query";
import {createBrowserRouter, RouterProvider} from 'react-router-dom';
import {MainLayout} from "./layouts/MainLayout.tsx";
import ViewComputationGraphPage from "./pages/ViewComputationGraph.tsx";
import ViewTensorPage from "./pages/ViewTensor.tsx";
import AnalyzeTextPage from "./pages/AnalyzeText.tsx";
import ViewModelPage from "./pages/ViewModel.tsx";
import HomePage from './pages/Home.tsx';
import AnalyzeAttentionPage from "./pages/AnalyzeAttention.tsx";

const queryClient = new QueryClient();


const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        path: '/home',
        element: <HomePage />,
      },
      {
        path: '/model',
        element: <ViewModelPage />,
      },
      {
        path: '/model/computation-graph',
        element: <ViewComputationGraphPage />,
      },
      {
        path: '/tensor/:tensorId',
        element: <ViewTensorPage />,
      },
      {
        path: '/analyze/global/attention',
        element: <AnalyzeAttentionPage />,
      },
      {
        path: '/analyze/local/text',
        element: <AnalyzeTextPage />,
      }
    ]
  },
]);


ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      {/*<App />*/}
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>,
)
