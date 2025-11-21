import { useState } from "react";
import { Filter, Settings, History, Trash2, ChevronDown } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
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
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isTortTypesOpen, setIsTortTypesOpen] = useState(false);
  const [isFiltersOpen, setIsFiltersOpen] = useState(false);

  const handleTortTypeToggle = (tortType: string) => {
    const newTypes = filters.tortTypes.includes(tortType)
      ? filters.tortTypes.filter((t) => t !== tortType)
      : [...filters.tortTypes, tortType];
    onFilterChange({ tortTypes: newTypes });
  };

  return (
    <aside className="w-80 min-w-[20rem] border-r bg-card overflow-y-auto">
      <ScrollArea className="h-full">
        <div className="p-4 space-y-6">
          {/* Filters Section */}
          <Card className="bg-background/60 backdrop-blur-sm border-border">
            <Collapsible open={isFiltersOpen} onOpenChange={setIsFiltersOpen}>
              <CardHeader className="pb-3">
                <CollapsibleTrigger asChild>
                  <div className="flex items-center justify-between cursor-pointer w-full select-none">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Filter className="h-4 w-4" />
                      Filters
                    </CardTitle>
                    <ChevronDown
                      className={`h-4 w-4 transition-transform duration-200 ${
                        isFiltersOpen ? "rotate-180" : ""
                      }`}
                    />
                  </div>
                </CollapsibleTrigger>
              </CardHeader>
              <CollapsibleContent>
                <CardContent className="space-y-4">
                  {/* Year Range */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label className="text-xs">Case Year Range</Label>
                      <span className="text-xs font-medium text-muted-foreground">
                        {filters.yearRange[0]} - {filters.yearRange[1]}
                      </span>
                    </div>
                    <Slider
                      min={1950}
                      max={2025}
                      step={1}
                      value={filters.yearRange}
                      onValueChange={(value) =>
                        onFilterChange({
                          yearRange: value as [number, number],
                        })
                      }
                      className="py-2"
                    />
                  </div>

                  <Separator />

                  {/* Tort Types */}
                  <Collapsible
                    open={isTortTypesOpen}
                    onOpenChange={setIsTortTypesOpen}
                    className="space-y-2"
                  >
                    <CollapsibleTrigger asChild>
                      <div className="flex items-center justify-between cursor-pointer w-full select-none">
                        <Label className="text-xs cursor-pointer">
                          Tort Types
                        </Label>
                        <ChevronDown
                          className={`h-4 w-4 transition-transform duration-200 ${
                            isTortTypesOpen ? "rotate-180" : ""
                          }`}
                        />
                      </div>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <ScrollArea className="h-48 border rounded-md p-2 mt-2">
                        <div className="space-y-2">
                          {tortTypeOptions.map((type) => (
                            <div
                              key={type}
                              className="flex items-center space-x-2"
                            >
                              <Checkbox
                                id={type}
                                checked={filters.tortTypes.includes(type)}
                                onCheckedChange={() =>
                                  handleTortTypeToggle(type)
                                }
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
                    </CollapsibleContent>
                  </Collapsible>
                </CardContent>
              </CollapsibleContent>
            </Collapsible>
          </Card>

          {/* Settings Section */}
          <Card className="bg-background/60 backdrop-blur-sm border-border">
            <Collapsible open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
              <CardHeader className="pb-3">
                <CollapsibleTrigger asChild>
                  <div className="flex items-center justify-between cursor-pointer w-full">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Settings className="h-4 w-4" />
                      Settings
                    </CardTitle>
                    <ChevronDown
                      className={`h-4 w-4 transition-transform duration-200 ${
                        isSettingsOpen ? "rotate-180" : ""
                      }`}
                    />
                  </div>
                </CollapsibleTrigger>
              </CardHeader>
              <CollapsibleContent>
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
                      Semantic Weight:{" "}
                      {(filters.semanticWeight * 100).toFixed(0)}%
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
                      Keyword: {((1 - filters.semanticWeight) * 100).toFixed(0)}
                      %
                    </p>
                  </div>
                </CardContent>
              </CollapsibleContent>
            </Collapsible>
          </Card>

          {/* Search History */}
          <Card className="bg-background/60 backdrop-blur-sm border-border">
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
                          {query.substring(0, 40)}
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
            className="w-full bg-destructive/80 hover:bg-destructive/90 transition-all duration-300 hover:scale-105 backdrop-blur-sm"
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
