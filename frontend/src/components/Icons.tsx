import type { ReactNode, SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;
const base = (children: ReactNode, props: P) => (
  <svg aria-hidden="true" viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" {...props}>
    {children}
  </svg>
);

export const IconHome = (p: P) => base(<path d="M3 11l9-7 9 7v9a1 1 0 01-1 1h-5v-6H9v6H4a1 1 0 01-1-1z" />, p);
export const IconLayers = (p: P) => base(<><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5" /><path d="M3 17.5l9 5 9-5" opacity=".6" /></>, p);
export const IconChip = (p: P) => base(<><rect x="6" y="6" width="12" height="12" rx="2" /><rect x="9.5" y="9.5" width="5" height="5" rx="1" /><path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" /></>, p);
export const IconFile = (p: P) => base(<><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z" /><path d="M14 3v5h5M9 13h6M9 17h6" /></>, p);
export const IconInfo = (p: P) => base(<><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></>, p);
export const IconGithub = (p: P) => base(<path d="M9 19c-5 1.500-5-2.500-7-3m14 6v-3.870a3.370 3.370 0 00-.940-2.610c3.140-.350 6.440-1.540 6.440-7A5.440 5.440 0 0020 4.770 5.070 5.070 0 0019.910 1S18.730.650 16 2.480a13.380 13.380 0 00-7 0C6.270.650 5.090 1 5.090 1A5.070 5.070 0 005 4.770a5.440 5.440 0 00-1.500 3.780c0 5.420 3.300 6.610 6.440 7A3.370 3.370 0 009 18.130V22" />, p);
export const IconBolt = (p: P) => base(<path d="M13 2L4 14h7l-1 8 9-12h-7z" />, p);
export const IconShield = (p: P) => base(<><path d="M12 3l8 3v6c0 4.500-3.200 8-8 9-4.800-1-8-4.500-8-9V6z" /><path d="M8.500 12l2.500 2.500L15.500 10" /></>, p);
export const IconMonitor = (p: P) => base(<><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>, p);
export const IconImage = (p: P) => base(<><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="9" cy="10" r="1.600" /><path d="M21 16l-5-5-8 9" /></>, p);
export const IconVideo = (p: P) => base(<><rect x="3" y="6" width="13" height="12" rx="2" /><path d="M16 10l5-3v10l-5-3z" /></>, p);
export const IconUpload = (p: P) => base(<><path d="M7 17a4 4 0 01-.8-7.900A6 6 0 0117.700 8 4.500 4.500 0 0117.500 17" /><path d="M12 12v8M9 15l3-3 3 3" /></>, p);
export const IconFolder = (p: P) => base(<path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />, p);
export const IconScan = (p: P) => base(<><path d="M4 8V5a1 1 0 011-1h3M16 4h3a1 1 0 011 1v3M20 16v3a1 1 0 01-1 1h-3M8 20H5a1 1 0 01-1-1v-3" /><circle cx="12" cy="12" r="3.500" /></>, p);
export const IconChart = (p: P) => base(<><path d="M5 20V10M12 20V4M19 20v-7" /></>, p);
export const IconDownload = (p: P) => base(<><path d="M12 4v11M8 11l4 4 4-4" /><path d="M5 20h14" /></>, p);
export const IconTrash = (p: P) => base(<><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" /><path d="M10 11v6M14 11v6" /></>, p);
export const IconArrowLeft = (p: P) => base(<path d="M19 12H5M11 6l-6 6 6 6" />, p);
export const IconWarning = (p: P) => base(<><path d="M12 3l10 18H2z" /><path d="M12 10v5M12 18h.01" /></>, p);
export const IconCheckCircle = (p: P) => base(<><circle cx="12" cy="12" r="9" /><path d="M8 12.500l2.800 2.800L16 9.500" /></>, p);
export const IconHelp = (p: P) => base(<><circle cx="12" cy="12" r="9" /><path d="M9.500 9.500a2.500 2.500 0 114 2c-.8.700-1.500 1.100-1.500 2.200M12 17h.01" /></>, p);
export const IconClock = (p: P) => base(<><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>, p);
export const IconCalendar = (p: P) => base(<><rect x="3" y="5" width="18" height="16" rx="2" /><path d="M3 10h18M8 3v4M16 3v4" /></>, p);
export const IconMenu = (p: P) => base(<path d="M4 7h16M4 12h16M4 17h16" />, p);
export const IconClose = (p: P) => base(<path d="M6 6l12 12M18 6L6 18" />, p);
export const IconGauge = (p: P) => base(<><path d="M4 16a8 8 0 1116 0" /><path d="M12 16l4-5" /></>, p);
export const IconUser = (p: P) => base(<><circle cx="12" cy="8" r="3.500" /><path d="M5 20a7 7 0 0114 0" /></>, p);
export const IconStack = (p: P) => base(<><rect x="4" y="4" width="16" height="16" rx="2" /><path d="M4 10h16M10 4v16" /></>, p);
export const IconDatabase = (p: P) => base(<><ellipse cx="12" cy="6" rx="7" ry="3" /><path d="M5 6v6c0 1.700 3.100 3 7 3s7-1.300 7-3V6M5 12v6c0 1.700 3.100 3 7 3s7-1.300 7-3v-6" /></>, p);
export const IconActivity = (p: P) => base(<path d="M3 12h4l2.500-6 4 12 2.500-6H21" />, p);
export const IconArrowRight = (p: P) => base(<path d="M5 12h14M13 6l6 6-6 6" />, p);
export const IconChevronLeft = (p: P) => base(<path d="M15 5l-7 7 7 7" />, p);
export const IconChevronRight = (p: P) => base(<path d="M9 5l7 7-7 7" />, p);
export const IconEye = (p: P) => base(<><path d="M2 12s3.500-7 10-7 10 7 10 7-3.500 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></>, p);
