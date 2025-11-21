import { Hospital, Shield, Newspaper, Briefcase, Zap, Car } from "lucide-react";
import { Button } from "@/components/ui/button";

interface QuerySuggestionsProps {
  onSuggestionClick: (query: string) => void;
}

const suggestions = [
  {
    icon: Hospital,
    text: "Cases on medical negligence",
    query: "Cases on medical negligence compensation",
  },
  {
    icon: Shield,
    text: "Police custody torture and damages",
    query: "Police custody torture and damages",
  },
  {
    icon: Newspaper,
    text: "Defamation by media houses",
    query: "Defamation by media houses",
  },
  {
    icon: Briefcase,
    text: "Vicarious liability of employers",
    query: "Vicarious liability of employers",
  },
  {
    icon: Zap,
    text: "Strict liability in hazardous activities",
    query: "Strict liability in hazardous activities",
  },
  {
    icon: Car,
    text: "Motor vehicle accident compensation",
    query: "Motor vehicle accident compensation",
  },
];

const QuerySuggestions = ({ onSuggestionClick }: QuerySuggestionsProps) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 space-y-6">
      <div className="text-center space-y-2">
        <h3 className="text-2xl font-bold text-foreground">
          Try these example queries
        </h3>
        <p className="text-muted-foreground">
          {/* Click any suggestion to start searching for relevant case laws */}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-w-5xl w-full">
        {suggestions.map((suggestion, index) => {
          const Icon = suggestion.icon;
          return (
            <Button
              key={index}
              variant="outline"
              className="h-auto py-4 px-4 justify-start text-left hover:bg-primary/10 hover:border-primary transition-colors group"
              onClick={() => onSuggestionClick(suggestion.query)}
            >
              <div className="p-2 rounded-lg bg-primary/10 backdrop-blur-sm border border-primary/20 mr-3 group-hover:bg-primary/20 transition-colors">
                <Icon className="h-5 w-5 text-primary" />
              </div>
              <span className="text-sm">{suggestion.text}</span>
            </Button>
          );
        })}
      </div>
    </div>
  );
};

export default QuerySuggestions;
