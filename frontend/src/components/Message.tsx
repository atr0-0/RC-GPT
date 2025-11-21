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
              ? "bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800"
              : isError
              ? "bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-800"
              : "bg-muted/50 border-border"
          } overflow-hidden`}
        >
          <CardContent className="p-4">
            <div className="prose prose-sm dark:prose-invert max-w-none break-words">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
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
            <CollapsibleContent className="space-y-3 mt-3">
              {message.sources.map((source, index) => {
                const badge = getSectionBadge(source.section_type);
                return (
                  <Card
                    key={index}
                    className="bg-gradient-to-br from-yellow-50 to-orange-50 dark:from-yellow-950 dark:to-orange-950 border-yellow-200 dark:border-yellow-800"
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between gap-2 flex-wrap">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge variant="outline" className="text-xs">
                              Source {index + 1}
                            </Badge>
                            <Badge className="bg-green-500 hover:bg-green-600 text-white text-xs">
                              {source.confidence}
                            </Badge>
                          </div>
                          <CardTitle className="text-sm font-semibold">
                            {source.case_name}
                          </CardTitle>
                        </div>
                      </div>

                      {source.citation && source.citation !== "N/A" && (
                        <Badge
                          variant="secondary"
                          className="mt-2 w-fit text-xs"
                        >
                          📖 {source.citation}
                        </Badge>
                      )}

                      <Badge
                        className="mt-2 w-fit text-xs text-white"
                        style={{ backgroundColor: badge.color }}
                      >
                        {badge.label}
                      </Badge>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {source.excerpt}
                      </p>
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
