import React from "react";
import { Line } from "react-chartjs-2";
import "chartjs-plugin-dragdata";




const DraggableGraph = ({ dataSetCurrent, setDataCurrent, dataSetSimp, dataSetOriginal, updateData, lineColorCurr, lineColorSimp, lineColorOrg }) => {
  if (!dataSetCurrent){
    dataSetCurrent = [];
  }
  if (!dataSetSimp){
    dataSetSimp = [];
  }
  if (!dataSetOriginal){
    dataSetOriginal = [];

  }
  const rawMax = dataSetOriginal.length > 0 ? Math.max(...dataSetOriginal): 1;
  const rawMin =dataSetOriginal.length > 0 ? Math.min(...dataSetOriginal): -1;
  const niceMax = Math.ceil(rawMax);
  const niceMin = Math.floor(rawMin);


  const data_label = Array.from({ length: dataSetOriginal.length }, (_, i) => i);
  const state = {
    dataSet: [dataSetCurrent, dataSetSimp, dataSetOriginal],
    labels: data_label,
    options: {
      tooltips: { enabled: true },
      scales: {
        x: [
          {
            gridLines: { display: true, color: "grey" },
            ticks: {
              fontColor: "#3C3C3C",
              fontSize: 14,
              callback: function (value, index) {
                const step_size = 2
                return index % step_size == 0 ? value : null;
              }
            }
          }
        ],
        y: [
          {
            scaleLabel: {
              display: true,
              labelString: "Domain Spesific Y label",
              fontSize: 14
            },
            ticks: {
              display: true,
              suggestedMin: niceMin,
              suggestedMax: niceMax,
              stepSize: 1,
              maxTicksLimit: 10,
              fontColor: "#000000",
              padding: 30,
              callback: function (value, index) {
                const step_size = 10
                return index % step_size === 0 ? value : null;
              }
            },
            gridLines: {
              display: true,
              offsetGridLines: false,
              color: "#3C3C3C",
              tickMarkLength: 4
            }
          }
        ]
      },
      legend: {
        display: true,

      },
      dragData: true,
      dragOptions: {
        showTooltip: true
      },
      dragDataRound: 1,
      onDragStart: function (e) {
        //console.log("Start:", e);
      },
      onDrag: function (e, datasetIndex, index, value) {
        //console.log("Drag:", datasetIndex, index, value);
      },
      onDragEnd: function (e, datasetIndex, index, value) {
        //console.log("Drag End:", state.dataSet);
        const newDataSet = state.dataSet[0];
        newDataSet[index] = value;
        updateData([...newDataSet], setDataCurrent);

      }.bind(this)
    }
  };

  //console.log("RENDER");
  const data = {
    labels: state.labels,
    datasets: [
      {
        label: "Interactive",
        data: state.dataSet[0],
        lineTension: 0,
        borderColor: lineColorCurr,
        borderWidth: 5,
        pointRadius: 7,//7
        pointHoverRadius: 12,
        pointBackgroundColor: "black",
        pointBorderWidth: 0,
        spanGaps: false,
        dragData: true,
        fill: false
      },


      {
        label: "Simplification",
        data: state.dataSet[1],
        lineTension: 0,
        borderColor: lineColorSimp,
        borderWidth: 5,
        pointRadius: 0,
        pointHoverRadius: 1,
        pointBackgroundColor: lineColorSimp,
        pointBorderWidth: 0,
        spanGaps: false,
        dragData: false,
        fill: false,
        borderDash: [3, 6]

      },
      {
        label: "Prototype",
        data: state.dataSet[2],
        lineTension: 0,
        borderColor: lineColorOrg,
        borderWidth: 5,
        pointRadius: 1,
        pointHoverRadius: 1,
        pointBackgroundColor: lineColorOrg,
        pointBorderWidth: 0,
        spanGaps: false,
        dragData: false,
        fill: false

      }
    ]
  };
  return (
    <div>
      <Line data={data} options={state.options} />
    </div>
  );
};

export default DraggableGraph;
