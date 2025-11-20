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

const Landing = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Scale className="h-8 w-8 text-primary" />
            <span className="text-2xl font-bold text-foreground">
              CaseLawGPT
            </span>
          </div>
          <Link to="/chat">
            <Button>Get Started</Button>
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-accent/5" />
        <div className="container mx-auto px-4 py-24 md:py-32 relative">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 text-foreground">
              AI Legal Research for Indian Tort Law
            </h1>
            <p className="text-xl text-muted-foreground mb-8">
              Access 690+ Supreme Court tort law cases instantly. Get AI-powered
              insights, precise citations, and relevant case law to strengthen
              your legal arguments.
            </p>
            <div className="flex gap-4 justify-center">
              <Link to="/chat">
                <Button size="lg" className="bg-primary hover:bg-primary/90">
                  Start Researching
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-foreground">
              Why Choose CaseLawGPT?
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Powerful AI capabilities designed specifically for legal
              professionals researching Indian tort law cases.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <Database className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  690+ Supreme Court Cases
                </h3>
                <p className="text-muted-foreground">
                  Comprehensive database of Indian Supreme Court tort law
                  judgments from 1950 to 2025.
                </p>
              </CardContent>
            </Card>

            <Card className="border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <Zap className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  Hybrid Search
                </h3>
                <p className="text-muted-foreground">
                  Combines semantic understanding and keyword matching for
                  precise, relevant results.
                </p>
              </CardContent>
            </Card>

            <Card className="border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <BookOpen className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  Precise Citations
                </h3>
                <p className="text-muted-foreground">
                  Get exact case citations, excerpts, and confidence scores for
                  every source.
                </p>
              </CardContent>
            </Card>

            <Card className="border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <MessageSquare className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  Natural Conversation
                </h3>
                <p className="text-muted-foreground">
                  Ask questions in plain language - no legal jargon required.
                </p>
              </CardContent>
            </Card>

            <Card className="border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <FileText className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-semibold mb-2 text-foreground">
                  Advanced Filtering
                </h3>
                <p className="text-muted-foreground">
                  Filter by year range, tort types, and adjust search parameters
                  for targeted research.
                </p>
              </CardContent>
            </Card>

            <Card className="border-border hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <Shield className="h-12 w-12 text-primary mb-4" />
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
      <section className="py-20">
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
            <div className="text-center space-y-3">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                <Users className="h-8 w-8 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">For Lawyers</h3>
              <p className="text-muted-foreground">
                Find relevant precedents quickly to strengthen your arguments
                and build compelling cases.
              </p>
            </div>

            <div className="text-center space-y-3">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                <BookOpen className="h-8 w-8 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">For Researchers</h3>
              <p className="text-muted-foreground">
                Explore tort law evolution and analyze judicial patterns across
                decades.
              </p>
            </div>

            <div className="text-center space-y-3">
              <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto">
                <FileText className="h-8 w-8 text-primary" />
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
      <section className="py-20">
        <div className="container mx-auto px-4">
          <Card className="bg-gradient-to-br from-primary to-primary/80 border-0">
            <CardContent className="p-12 text-center">
              <h2 className="text-3xl md:text-4xl font-bold mb-4 text-primary-foreground">
                Ready to Transform Your Legal Research?
              </h2>
              <p className="text-lg text-primary-foreground/90 mb-8 max-w-2xl mx-auto">
                Join lawyers across India who are using CaseLawGPT to find
                relevant precedents faster.
              </p>
              <Link to="/chat">
                <Button
                  size="lg"
                  variant="secondary"
                  className="bg-background hover:bg-background/90 text-foreground"
                >
                  Start Using CaseLawGPT
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>
            &copy; 2025 CaseLawGPT. AI Legal Research Assistant for Indian Tort
            Law.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
