


export interface SidebarItem {
  text: string;
  path: string;
  children?: SidebarItem[];
}


export interface SidebarGroup {
  title: string;
  items: SidebarItem[];
}


export interface SidebarConfig {
  groups: SidebarGroup[];
}


export const MAIN_SIDEBAR: SidebarConfig = {
  groups: [
    {
      title: 'General',
      items: [
        {
          text: 'Home',
          path: '/home',
        },
        {
          text: 'Projects',
          path: '/projects',
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
          path: '/analyze/global/attention/head-clustering',
        },
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
};
