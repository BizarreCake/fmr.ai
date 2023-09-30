


export interface SidebarItem {
  text: string;
  path: string;
  children?: SidebarItem[];
}


export interface SidebarGroup {
  title?: string;
  items: SidebarItem[];
}


export interface SidebarConfig {
  rootPath?: string;
  groups: SidebarGroup[];
}


export const MAIN_SIDEBAR: SidebarConfig = {
  groups: [
    {
      items: [
        {
          text: 'Home',
          path: '/home',
        },
        {
          text: 'Projects',
          path: '/projects',
        },
      ],
    },
  ],
};

export const PROJECT_SIDEBAR = {
  groups: [
    {
      title: 'General',
      items: [
        {
          text: 'Agents',
          path: '/agents',
        },
        {
          text: 'Model Graph',
          path: '/model/computation-graph',
        },
      ],
    },
    {
      title: 'Global',
      items: [
        {
          text: 'Attention Heads',
          path: '/analysis/global/attention/head-clustering',
        },
        {
          text: 'Key-Value Memories',
          path: '/analysis/global/key-value-memories',
        }
      ]
    },
    {
      title: 'Local',
      items: [
        {
          text: 'Analyze Text',
          path: '/analyze/local/text',
        },
      ]
    }
  ],
}
