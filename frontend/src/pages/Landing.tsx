import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Scale,
  Shield,
  Zap,
  Users,
  FileText,
  MessageSquare,
  Database,
  BookOpen,
} from "lucide-react";
import { Link } from "react-router-dom";
import Prism from "@/components/Prism";
import { useTheme } from "next-themes";
import GlassIcon from "@/components/GlassIcon";

const Landing = () => {
  const { theme, resolvedTheme } = useTheme();
  const isDark = theme === "dark" || resolvedTheme === "dark";

  return (
    <div className="relative min-h-screen bg-background">
      {/* Animated Background Prism */}
      <div className="fixed inset-0 z-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute inset-0 opacity-60">
          <Prism
            animationType="rotate"
            timeScale={0.5}
            height={3.5}
            baseWidth={5.5}
            scale={3.6}
            hueShift={0}
            colorFrequency={1}
            noise={0.5}
            glow={1}
          />
        </div>
      </div>
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Scale className="h-8 w-8 text-primary" />
            <span className="text-2xl font-bold text-foreground">
              RC-GPT
            </span>
          </div>
          <Link to="/chat">
            <Button className="bg-primary/80 hover:bg-primary/90 transition-all duration-300 hover:scale-110 backdrop-blur-sm">
              Get Started
            </Button>
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative">
        <div className="absolute inset-0 bg-gradient-to-br from-background/40 via-background/30 to-background/40 backdrop-blur-[1px]" />
        <div className="container mx-auto px-4 py-24 md:py-32 relative z-10">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 text-foreground">
              India's First Case Law Retrieval AI Agent
            </h1>
            <p className="text-xl text-muted-foreground mb-8">
              Get AI-powered
              insights, precise citations, and relevant case law to strengthen
              your legal arguments.
            </p>
            <div className="flex gap-4 justify-center">
              <Link to="/chat">
                <Button
                  size="lg"
                  className="bg-primary/80 hover:bg-primary/90 transition-all duration-300 hover:scale-110 backdrop-blur-sm"
                >
                  Try RC-GPT
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 relative z-10">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-foreground">
              Why Choose RC-GPT?
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Powerful AI capabilities designed specifically for legal
              professionals researching Indian tort law cases.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="bg-background/60 backdrop-blur-sm border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="mb-4">
                  <GlassIcon
                    icon={<Database className="h-8 w-8 text-primary" />}
                  />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  690+ Supreme Court Cases
                </h3>
                <p className="text-muted-foreground">
                  Comprehensive database of Indian Supreme Court tort law
                  judgments from 1950 to 2025.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-background/60 backdrop-blur-sm border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="mb-4">
                  <GlassIcon icon={<Zap className="h-8 w-8 text-primary" />} />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  Hybrid Search
                </h3>
                <p className="text-muted-foreground">
                  Combines semantic understanding and keyword matching for
                  precise, relevant results.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-background/60 backdrop-blur-sm border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="mb-4">
                  <GlassIcon
                    icon={<BookOpen className="h-8 w-8 text-primary" />}
                  />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  Precise Citations
                </h3>
                <p className="text-muted-foreground">
                  Get exact case citations, excerpts, and confidence scores for
                  every source.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-background/60 backdrop-blur-sm border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="mb-4">
                  <GlassIcon
                    icon={<MessageSquare className="h-8 w-8 text-primary" />}
                  />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  Natural Conversation
                </h3>
                <p className="text-muted-foreground">
                  Ask questions in plain language - no legal jargon required.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-background/60 backdrop-blur-sm border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="mb-4">
                  <GlassIcon
                    icon={<FileText className="h-8 w-8 text-primary" />}
                  />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  Advanced Filtering
                </h3>
                <p className="text-muted-foreground">
                  Filter by year range, tort types, and adjust search parameters
                  for targeted research.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-background/60 backdrop-blur-sm border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="mb-4">
                  <GlassIcon
                    icon={<Shield className="h-8 w-8 text-primary" />}
                  />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  AI-Powered Analysis
                </h3>
                <p className="text-muted-foreground">
                  Powered by Google Gemini for intelligent case law analysis and
                  summaries.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section className="py-20 relative z-10">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-foreground">
              Perfect for Every Legal Need
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Whether you're building a case, researching precedents, or
              studying tort law.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <div className="text-center space-y-3 flex flex-col items-center">
              <div className="mb-2">
                <GlassIcon icon={<Users className="h-8 w-8 text-primary" />} />
              </div>
              <h3 className="text-xl font-semibold">For Lawyers</h3>
              <p className="text-muted-foreground">
                Find relevant precedents quickly to strengthen your arguments
                and build compelling cases.
              </p>
            </div>

            <div className="text-center space-y-3 flex flex-col items-center">
              <div className="mb-2">
                <GlassIcon
                  icon={<BookOpen className="h-8 w-8 text-primary" />}
                />
              </div>
              <h3 className="text-xl font-semibold">For Researchers</h3>
              <p className="text-muted-foreground">
                Explore tort law evolution and analyze judicial patterns across
                decades.
              </p>
            </div>

            <div className="text-center space-y-3 flex flex-col items-center">
              <div className="mb-2">
                <GlassIcon
                  icon={<FileText className="h-8 w-8 text-primary" />}
                />
              </div>
              <h3 className="text-xl font-semibold">For Students</h3>
              <p className="text-muted-foreground">
                Study real Supreme Court cases and understand tort law concepts
                through practical examples.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 relative z-10">
        <div className="container mx-auto px-4">
          <Card className="bg-background/60 backdrop-blur-sm border-border">
            <CardContent className="p-12 text-center">
              <h2 className="text-3xl md:text-4xl font-bold mb-4 text-foreground">
                Ready to Transform Your Legal Research?
              </h2>
              <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
                Join lawyers across India who are using RC-GPT to find
                relevant precedents faster.
              </p>
              <Link to="/chat">
                <Button
                  size="lg"
                  className="bg-primary/60 hover:bg-primary/70 transition-all duration-300 hover:scale-110 backdrop-blur-sm"
                >
                  Start Using RC-GPT
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 relative z-10">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>
            &copy; 2025 RC-GPT. AI Legal Research Assistant for Indian Tort
            Law.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
