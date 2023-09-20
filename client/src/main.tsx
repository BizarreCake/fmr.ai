import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import {QueryClient, QueryClientProvider} from "react-query";
import {createBrowserRouter, RouterProvider} from 'react-router-dom';
import {MainLayout} from "./layouts/MainLayout.tsx";
import ViewComputationGraphPage from "./pages/ViewComputationGraph.tsx";
import ViewTensorPage from "./pages/ViewTensor.tsx";
import AnalyzeTextPage from "./pages/AnalyzeText.tsx";
import ModelHomePage from "./pages/ModelHome.tsx";
import HomePage from './pages/Home.tsx';
import AttentionHomePage from "./pages/AttentionHome.tsx";
import AttentionClusteringPage from "./pages/AttentionClustering.tsx";
import ViewAttentionClusteringPage from "./pages/ViewAttentionClustering.tsx";

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
        element: <ModelHomePage />,
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
        element: <AttentionHomePage />,
      },
      {
        path: '/analyze/global/attention/head-clustering',
        element: <AttentionClusteringPage />,
      },

      {
        path: '/analyze/global/attention/head-clustering/:key',
        element: <ViewAttentionClusteringPage />,
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
