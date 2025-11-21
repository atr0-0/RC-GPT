import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Scale, Send, Menu, Loader } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { processQuery, QueryRequest, Source } from "@/services/api";
import Message from "@/components/Message";
import Sidebar from "@/components/Sidebar";
import QuerySuggestions from "@/components/QuerySuggestions";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import LightRays from "@/components/LightRays";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  searchType?: string;
  isError?: boolean;
}

interface Filters {
  yearRange: [number, number];
  tortTypes: string[];
  maxSources: number;
  semanticWeight: number;
}

const Chat = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [filters, setFilters] = useState<Filters>({
    yearRange: [1950, 2025],
    tortTypes: [],
    maxSources: 5,
    semanticWeight: 0.7,
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // StatsBar removed; no stats fetching

  // Process query mutation
  const queryMutation = useMutation({
    mutationFn: processQuery,
    onSuccess: (data) => {
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        searchType: data.search_type,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    },
    onError: (error: any) => {
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: `Error: ${error.message || "Failed to process query"}`,
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
      toast({
        title: "Error",
        description: "Failed to process your query. Please try again.",
        variant: "destructive",
      });
    },
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (query?: string) => {
    const queryText = query || input.trim();
    if (!queryText || queryMutation.isPending) return;

    setInput("");

    // Add user message
    const userMessage: ChatMessage = {
      role: "user",
      content: queryText,
    };
    setMessages((prev) => [...prev, userMessage]);

    // Add to search history
    if (!searchHistory.includes(queryText)) {
      setSearchHistory((prev) => [...prev.slice(-9), queryText]);
    }

    // Call API
    const requestData: QueryRequest = {
      query: queryText,
      year_range: filters.yearRange,
      tort_types: filters.tortTypes.length > 0 ? filters.tortTypes : undefined,
      max_sources: filters.maxSources,
    };

    queryMutation.mutate(requestData);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    toast({
      title: "Chat cleared",
      description: "Your chat history has been cleared.",
    });
  };

  const handleFilterChange = (newFilters: Partial<Filters>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
  };

  const handleSuggestionClick = (query: string) => {
    handleSubmit(query);
  };

  const handleHistoryClick = (query: string) => {
    handleSubmit(query);
  };

  // Check if we should show suggestions
  const hasSubstantiveAnswer = messages.some(
    (m) => m.role === "assistant" && m.sources && m.sources.length > 0
  );

  return (
    <div className="flex flex-col h-screen bg-background relative">
      {/* Header */}
      <header className="border-b bg-card relative z-10">
        <div className="flex items-center gap-4 p-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <Menu className="h-5 w-5" />
          </Button>
          <div className="flex items-center gap-2">
            <Scale className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-bold text-foreground">RC-GPT</h1>
          </div>
          <p className="text-sm text-muted-foreground hidden md:block">
            AI Legal Research Assistant for Indian Tort Law
          </p>
        </div>
      </header>

      {/* StatsBar removed */}

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden relative z-10">
        {/* Sidebar */}
        {sidebarOpen && (
          <Sidebar
            filters={filters}
            onFilterChange={handleFilterChange}
            searchHistory={searchHistory}
            onClearChat={handleClearChat}
            onHistoryClick={handleHistoryClick}
          />
        )}

        {/* Chat Area */}
        <div className="flex-1 flex flex-col min-w-0 relative">
          {/* Animated Background LightRays */}
          <div className="absolute inset-0 z-0 w-full h-full overflow-hidden pointer-events-none">
            <div className="absolute inset-0 opacity-30">
              <LightRays
                raysOrigin="top-center"
                raysColor="#00ffff"
                raysSpeed={1.5}
                lightSpread={0.8}
                rayLength={1.2}
                followMouse={true}
                mouseInfluence={0.1}
                noiseAmount={0.1}
                distortion={0.05}
                className="custom-rays"
              />
            </div>
          </div>

          {/* Messages */}
          <ScrollArea className="flex-1 p-4 relative z-10">
            {!hasSubstantiveAnswer && messages.length === 0 && (
              <QuerySuggestions onSuggestionClick={handleSuggestionClick} />
            )}

            {messages.map((message, index) => (
              <Message key={index} message={message} />
            ))}

            {queryMutation.isPending && (
              <div className="flex items-center gap-2 text-muted-foreground py-4">
                <Loader className="h-6 w-6 animate-spin" />
                <span>Searching case law database...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </ScrollArea>

          {/* Input */}
          <div className="border-t bg-card p-4 relative z-10">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSubmit();
              }}
              className="flex gap-2"
            >
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about tort law cases..."
                disabled={queryMutation.isPending}
                className="flex-1"
              />
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || queryMutation.isPending}
              >
                <Send className="h-4 w-4" />
              </Button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
