import type { Metadata } from "next";
import { BuildingSanjayaArticlePage } from "@/components/blog/building-sanjaya-page";

export const metadata: Metadata = {
  title: "Building Sanjaya",
  description: "Building Sanjaya: An RLM agent that programs its way through videos and images.",
};

export default function BuildingSanjayaPage() {
  return <BuildingSanjayaArticlePage />;
}
