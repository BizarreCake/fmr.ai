


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
          text: 'Model',
          path: '/model',
          children: [
            {
              text: 'Computation Graph',
              path: '/model/computation-graph',
            },
          ],
        },
      ],
    },
    {
      title: 'Instance',
      items: [
        {
          text: 'Analyze Text',
          path: '/analyze/text',
        },
      ]
    }
  ],
};
