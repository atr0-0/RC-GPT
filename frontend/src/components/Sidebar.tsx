import { Info, Filter, Settings, History, Trash2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";

interface Filters {
  yearRange: [number, number];
  tortTypes: string[];
  maxSources: number;
  semanticWeight: number;
}

interface SidebarProps {
  filters: Filters;
  onFilterChange: (filters: Partial<Filters>) => void;
  searchHistory: string[];
  onClearChat: () => void;
  onHistoryClick: (query: string) => void;
}

const tortTypeOptions = [
  "negligence",
  "defamation",
  "trespass",
  "nuisance",
  "assault",
  "battery",
  "false_imprisonment",
  "malicious_prosecution",
  "conversion",
  "strict_liability",
  "vicarious_liability",
];

const Sidebar = ({
  filters,
  onFilterChange,
  searchHistory,
  onClearChat,
  onHistoryClick,
}: SidebarProps) => {
  const handleTortTypeToggle = (tortType: string) => {
    const newTypes = filters.tortTypes.includes(tortType)
      ? filters.tortTypes.filter((t) => t !== tortType)
      : [...filters.tortTypes, tortType];
    onFilterChange({ tortTypes: newTypes });
  };

  return (
    <aside className="w-80 border-r bg-card overflow-y-auto">
      <ScrollArea className="h-full">
        <div className="p-4 space-y-6">
          {/* About Section */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Info className="h-4 w-4" />
                About
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-2">
              <p className="text-muted-foreground">
                <strong className="text-foreground">CaseLawGPT</strong> helps
                lawyers find relevant Supreme Court tort law judgments.
              </p>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    🔍
                  </Badge>
                  <span className="text-muted-foreground">Hybrid search</span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    💬
                  </Badge>
                  <span className="text-muted-foreground">
                    Conversational memory
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    📚
                  </Badge>
                  <span className="text-muted-foreground">
                    690+ case database
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    🎯
                  </Badge>
                  <span className="text-muted-foreground">
                    Precise citations
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    🎚️
                  </Badge>
                  <span className="text-muted-foreground">
                    Advanced filtering
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Filters Section */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Filter className="h-4 w-4" />
                Filters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Year Range */}
              <div className="space-y-2">
                <Label className="text-xs">📅 Case Year Range</Label>
                <div className="text-sm font-semibold text-center py-1">
                  {filters.yearRange[0]} - {filters.yearRange[1]}
                </div>
                <div className="space-y-3">
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">
                      Start Year: {filters.yearRange[0]}
                    </Label>
                    <Slider
                      min={1950}
                      max={2025}
                      step={1}
                      value={[filters.yearRange[0]]}
                      onValueChange={(value) =>
                        onFilterChange({
                          yearRange: [value[0], filters.yearRange[1]],
                        })
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">
                      End Year: {filters.yearRange[1]}
                    </Label>
                    <Slider
                      min={1950}
                      max={2025}
                      step={1}
                      value={[filters.yearRange[1]]}
                      onValueChange={(value) =>
                        onFilterChange({
                          yearRange: [filters.yearRange[0], value[0]],
                        })
                      }
                    />
                  </div>
                </div>
              </div>

              <Separator />

              {/* Tort Types */}
              <div className="space-y-2">
                <Label className="text-xs">⚖️ Tort Types</Label>
                <ScrollArea className="h-48 border rounded-md p-2">
                  <div className="space-y-2">
                    {tortTypeOptions.map((type) => (
                      <div key={type} className="flex items-center space-x-2">
                        <Checkbox
                          id={type}
                          checked={filters.tortTypes.includes(type)}
                          onCheckedChange={() => handleTortTypeToggle(type)}
                        />
                        <label
                          htmlFor={type}
                          className="text-xs font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                        >
                          {type.replace(/_/g, " ")}
                        </label>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </CardContent>
          </Card>

          {/* Settings Section */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Max Sources */}
              <div className="space-y-2">
                <Label htmlFor="maxSources" className="text-xs">
                  Max Sources
                </Label>
                <Input
                  id="maxSources"
                  type="number"
                  min={1}
                  max={10}
                  value={filters.maxSources}
                  onChange={(e) =>
                    onFilterChange({
                      maxSources: parseInt(e.target.value) || 5,
                    })
                  }
                  className="h-8"
                />
              </div>

              <Separator />

              {/* Semantic Weight */}
              <div className="space-y-2">
                <Label className="text-xs">
                  Semantic Weight: {(filters.semanticWeight * 100).toFixed(0)}%
                </Label>
                <Slider
                  min={0}
                  max={100}
                  step={10}
                  value={[filters.semanticWeight * 100]}
                  onValueChange={(value) =>
                    onFilterChange({ semanticWeight: value[0] / 100 })
                  }
                />
                <p className="text-xs text-muted-foreground">
                  Keyword: {((1 - filters.semanticWeight) * 100).toFixed(0)}%
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Search History */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <History className="h-4 w-4" />
                Recent Searches
              </CardTitle>
            </CardHeader>
            <CardContent>
              {searchHistory.length > 0 ? (
                <div className="space-y-2">
                  {searchHistory
                    .slice(-5)
                    .reverse()
                    .map((query, index) => (
                      <Button
                        key={index}
                        variant="ghost"
                        className="w-full justify-start text-left h-auto py-2 px-2"
                        onClick={() => onHistoryClick(query)}
                      >
                        <span className="text-xs truncate">
                          ↩️ {query.substring(0, 40)}
                          {query.length > 40 ? "..." : ""}
                        </span>
                      </Button>
                    ))}
                </div>
              ) : (
                <p className="text-xs text-muted-foreground text-center py-4">
                  No recent searches
                </p>
              )}
            </CardContent>
          </Card>

          {/* Clear Button */}
          <Button
            variant="destructive"
            className="w-full"
            onClick={onClearChat}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Clear Chat History
          </Button>
        </div>
      </ScrollArea>
    </aside>
  );
};

export default Sidebar;
