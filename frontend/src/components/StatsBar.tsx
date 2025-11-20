import { FileText, Database, Calendar, Activity } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { StatsResponse } from "@/services/api";

interface StatsBarProps {
  stats: StatsResponse;
}

const StatsBar = ({ stats }: StatsBarProps) => {
  return (
    <div className="border-b bg-muted/30 px-4 py-3">
      <div className="container mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card className="border-border">
            <CardContent className="p-3 flex items-center gap-3">
              <FileText className="h-8 w-8 text-primary flex-shrink-0" />
              <div>
                <div className="text-2xl font-bold text-foreground">
                  {stats.total_cases}
                </div>
                <div className="text-xs text-muted-foreground">SC Cases</div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardContent className="p-3 flex items-center gap-3">
              <Database className="h-8 w-8 text-primary flex-shrink-0" />
              <div>
                <div className="text-2xl font-bold text-foreground">
                  {stats.total_chunks?.toLocaleString()}
                </div>
                <div className="text-xs text-muted-foreground">
                  Legal Chunks
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardContent className="p-3 flex items-center gap-3">
              <Calendar className="h-8 w-8 text-primary flex-shrink-0" />
              <div>
                <div className="text-2xl font-bold text-foreground">
                  {stats.year_range}
                </div>
                <div className="text-xs text-muted-foreground">Year Range</div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border">
            <CardContent className="p-3 flex items-center gap-3">
              <Activity className="h-8 w-8 text-primary flex-shrink-0" />
              <div>
                <div className="text-2xl font-bold text-foreground">
                  {stats.uptime}
                </div>
                <div className="text-xs text-muted-foreground">Uptime</div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default StatsBar;
