import { useState } from "react";
import { User, Bot, BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Source } from "@/services/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  searchType?: string;
  isError?: boolean;
}

interface MessageProps {
  message: Message;
}

const getSectionBadge = (sectionType: string) => {
  const badges = {
    facts: { label: "📋 Facts", color: "#3b82f6" },
    judgment: { label: "⚖️ Judgment", color: "#8b5cf6" },
    reasoning: { label: "💡 Reasoning", color: "#f59e0b" },
    general: { label: "📄 General", color: "#6b7280" },
  };
  return badges[sectionType as keyof typeof badges] || badges.general;
};

const Message = ({ message }: MessageProps) => {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === "user";
  const isError = message.isError;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-[85%] ${isUser ? "order-2" : "order-1"} min-w-0`}>
        <div className="flex items-start gap-3 mb-2">
          <Avatar className={`${isUser ? "order-2" : "order-1"} flex-shrink-0`}>
            <AvatarFallback
              className={
                isUser
                  ? "bg-blue-100 text-blue-700"
                  : "bg-purple-100 text-purple-700"
              }
            >
              {isUser ? (
                <User className="h-5 w-5" />
              ) : (
                <Bot className="h-5 w-5" />
              )}
            </AvatarFallback>
          </Avatar>
          <div
            className={`flex flex-col ${
              isUser ? "items-end order-1" : "items-start order-2"
            } flex-1`}
          >
            <span className="text-sm font-semibold text-foreground mb-1">
              {isUser ? "You" : "CaseLawGPT"}
            </span>
          </div>
        </div>

        <Card
          className={`${
            isUser
              ? "bg-primary text-primary-foreground border-primary"
              : isError
              ? "bg-destructive/10 border-destructive/20"
              : "bg-card border-border shadow-sm"
          } overflow-hidden`}
        >
          <CardContent className="p-4">
            <div className="prose prose-sm dark:prose-invert max-w-none break-words leading-relaxed">
              <ReactMarkdown 
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({node, ...props}) => <h1 className="text-xl font-bold mt-4 mb-2" {...props} />,
                  h2: ({node, ...props}) => <h2 className="text-lg font-semibold mt-3 mb-2" {...props} />,
                  h3: ({node, ...props}) => <h3 className="text-base font-semibold mt-2 mb-1" {...props} />,
                  ul: ({node, ...props}) => <ul className="list-disc pl-4 my-2 space-y-1" {...props} />,
                  ol: ({node, ...props}) => <ol className="list-decimal pl-4 my-2 space-y-1" {...props} />,
                  li: ({node, ...props}) => <li className="pl-1 marker:text-muted-foreground" {...props} />,
                  p: ({node, ...props}) => <p className="my-2" {...props} />,
                  blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-primary/50 pl-4 italic my-2 text-muted-foreground" {...props} />,
                  code: ({node, ...props}) => <code className="bg-muted px-1 py-0.5 rounded text-sm font-mono" {...props} />,
                }}
              >
                {message.content || ""}
              </ReactMarkdown>
            </div>
          </CardContent>
        </Card>

        {message.sources && message.sources.length > 0 && (
          <Collapsible
            open={showSources}
            onOpenChange={setShowSources}
            className="mt-3"
          >
            <CollapsibleTrigger asChild>
              <Button
                variant="outline"
                className="w-full justify-between"
                size="sm"
              >
                <div className="flex items-center gap-2">
                  <BookOpen className="h-4 w-4" />
                  <span>View {message.sources.length} source excerpts</span>
                </div>
                {showSources ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="space-y-4 mt-4">
              {message.sources.map((source, index) => {
                const badge = getSectionBadge(source.section_type);
                return (
                  <Card
                    key={index}
                    className="bg-card border-border shadow-sm hover:shadow-md transition-shadow duration-200"
                  >
                    <CardHeader className="pb-2 border-b bg-muted/30">
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="font-mono text-xs">
                            #{index + 1}
                          </Badge>
                          <Badge 
                            variant="secondary" 
                            className="text-xs font-medium"
                            style={{ 
                              backgroundColor: `${badge.color}20`, 
                              color: badge.color,
                              borderColor: `${badge.color}40`
                            }}
                          >
                            {badge.label}
                          </Badge>
                        </div>
                        <Badge className="bg-green-600/90 hover:bg-green-600 text-white text-xs font-mono">
                          {source.confidence} Match
                        </Badge>
                      </div>
                      
                      <CardTitle className="text-base font-bold leading-tight text-foreground">
                        {source.case_name}
                      </CardTitle>
                      
                      {source.citation && source.citation !== "N/A" && (
                        <div className="flex items-center gap-1.5 mt-2 text-sm text-muted-foreground font-medium">
                          <BookOpen className="h-3.5 w-3.5" />
                          <span>{source.citation}</span>
                        </div>
                      )}
                    </CardHeader>
                    
                    <CardContent className="pt-4">
                      <div className="relative">
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary/20 rounded-full" />
                        <p className="text-sm text-muted-foreground leading-relaxed pl-4 italic">
                          "{source.excerpt}"
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </CollapsibleContent>
          </Collapsible>
        )}

        {message.searchType && (
          <Badge variant="outline" className="mt-2 text-xs">
            🔬 {message.searchType}
          </Badge>
        )}
      </div>
    </div>
  );
};

export default Message;
