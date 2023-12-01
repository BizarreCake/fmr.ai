import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import {QueryClient, QueryClientProvider} from "react-query";
import {createBrowserRouter, Navigate, RouterProvider} from 'react-router-dom';
import {MainLayout} from "./layouts/MainLayout.tsx";
import ViewComputationGraphPage from "./pages/ViewComputationGraph.tsx";
import ViewTensorPage from "./pages/ViewTensor.tsx";
import AnalyzeTextPage from "./pages/AnalyzeText.tsx";
import ModelHomePage from "./pages/ModelHome.tsx";
import HomePage from './pages/Home.tsx';
import AttentionHomePage from "./pages/AttentionHome.tsx";
import AttentionClusteringPage from "./pages/AttentionClustering.tsx";
import ViewAttentionClusteringPage from "./pages/ViewAttentionClustering.tsx";
import ProjectsPage from "./pages/Projects.tsx";
import {ProjectLayout} from "./layouts/ProjectLayout.tsx";
import ProjectAgentsPage from "./pages/project/Agents.tsx";
import KeyValueMemoriesPage from "./pages/KeyValueMemories.tsx";
import {useParams} from "react-router";
import ProjectHomePage from "./pages/ProjectHome.tsx";
import {TempLayout} from "./layouts/TempLayout.tsx";
import {KVHomePage} from "./pages/kv/KVHome.tsx";

const queryClient = new QueryClient();


function buildMainRouter() {
  return createBrowserRouter([
    {
      path: '/',
      element: <MainLayout />,
      children: [
        {
          index: true,
          element: <Navigate to="/home" />,
        },
        {
          path: '/home',
          element: <HomePage />,
        },
        {
          path: '/projects',
          element: <ProjectsPage />,
        },
        {
          path: '/model',
          element: <ModelHomePage />,
        },
        {
          path: '/tensor/:tensorId',
          element: <ViewTensorPage />,
        },
      ]
    },
    {
      path: '/project/:projectId',
      element: <ProjectLayout />,
      children: [
        {
          index: true,
          element: <ProjectHomePage />,
        },
        {
          path: '/project/:projectId/agents',
          element: <ProjectAgentsPage />,
        },
        {
          path: '/project/:projectId/model/computation-graph',
          element: <ViewComputationGraphPage />,
        },
        {
          path: '/project/:projectId/analysis/global/attention',
          element: <AttentionHomePage />,
        },
        {
          path: '/project/:projectId/analysis/global/attention/head-clustering',
          element: <AttentionClusteringPage />,
        },

        {
          path: '/project/:projectId/analysis/global/attention/head-clustering/:key',
          element: <ViewAttentionClusteringPage />,
        },
        {
          path: '/project/:projectId/analysis/global/key-value-memories',
          element: <KeyValueMemoriesPage />,
        },
        {
          path: '/project/:projectId/analysis/local/text',
          element: <AnalyzeTextPage />,
        }
      ]
    },
  ]);
}


function buildTempKVRouter() {
  return createBrowserRouter([
    {
      path: '/',
      element: <TempLayout/>,
      children: [
        {
          index: true,
          element: <Navigate to="/home" />,
        },
        {
          path: '/home',
          element: <KVHomePage />,
        },
      ]
    },
  ]);
}


function buildRouter() {
  switch (import.meta.env.VITE_WORKFLOW_KIND) {
    case 'kv':
      return buildTempKVRouter();

    default:
      return buildMainRouter();
  }
}


const router = buildRouter();


ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      {/*<App />*/}
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>,
)
