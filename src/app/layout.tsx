import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VibeCheck - Real-time Emotion Analysis",
  description: "Modern emotion analysis dashboard built with Next.js, TypeScript, and Tailwind CSS. Features real-time facial expression recognition and psychological metrics.",
  keywords: ["Next.js", "TypeScript", "Tailwind CSS", "shadcn/ui", "Emotion Analysis", "React", "Computer Vision"],
  authors: [{ name: "Selma Haci" }],
  openGraph: {
    title: "VibeCheck",
    description: "Real-time emotion analysis and psychological metrics",
    url: "https://vibecheck.example.com",
    siteName: "VibeCheck",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "VibeCheck",
    description: "Real-time emotion analysis and psychological metrics",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
