import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { SystemConfigProvider } from "@/lib/SystemConfigContext";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SunShift - Solar Energy Forecasting",
  description: "AI-Powered Solar Energy Forecasting & Analytics Platform",
  appleWebApp: {
    capable: true,
    statusBarStyle: 'default',
    title: 'SunShift',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  viewportFit: 'cover',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0f172a' },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} antialiased`}
      >
        <SystemConfigProvider>
          {children}
        </SystemConfigProvider>
      </body>
    </html>
  );
}
