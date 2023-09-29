import { ChartData, ChartDataset } from "chart.js";
import { FuzzyVariableDistributionPart, FuzzyVariableI } from "./models/fuzzy-variable-distribution";

export class FuzzyService {
    static convertFuzzyVariableToChartData(fuzzyVariableI: FuzzyVariableI): ChartData<"line", { x: number, y: number }[], number> {
        const fuzzyVariableMap = fuzzyVariableI.getFuzzyVariableMap();
        const datasets: ChartDataset<"line", {
            x: number;
            y: number;
        }[]>[] = [];

        const colorsMap = fuzzyVariableI.getFuzzyVariableColorsMap();

        for (const [key, part] of Object.entries(fuzzyVariableMap)) {
            datasets.push(this.createChartDatasetFromFuzzyVariableDistributionPart(part, key, colorsMap[key]))
        }
        return {
            datasets
        }

    }

    private static createChartDatasetFromFuzzyVariableDistributionPart(part: FuzzyVariableDistributionPart, label: string, borderColor: string): ChartDataset<"line", {
        x: number;
        y: number;
    }[]> {

        const data: {
            x: number;
            y: number;
        }[] = [];

        if (part.isTypeTrapezoidal()) {
            const partTrapezoidal = part;
            if (partTrapezoidal.a != null) {
                data.push({ x: partTrapezoidal.a, y: 0 });
            }
            data.push({ x: partTrapezoidal.b, y: 1 });
            data.push({ x: partTrapezoidal.c, y: 1 });
            if (partTrapezoidal.d != null) {
                data.push({ x: partTrapezoidal.d, y: 0 });
            }
        } else if (part.isTypeTriangular()) {
            const partTriangular = part;
            if (partTriangular.a != null) {
                data.push({ x: partTriangular.a, y: 0 });
            }
            data.push({ x: partTriangular.b, y: 1 });
            if (partTriangular.c != null) {
                data.push({ x: partTriangular.c, y: 0 });
            }

        }

        return {
            // id: 1,
            label,
            data,
            borderColor
        }
    }
}
