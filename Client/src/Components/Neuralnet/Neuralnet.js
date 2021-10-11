import React, { useRef, useEffect, useState } from "react";
import { select } from "d3";
import "./Neuralnet.css";

class Layer {
  constructor(index, padding = 1, stride = 1, channel = 3) {
    this.padding = padding;
    this.stride = stride;
    this.channel = channel;
    this.index = index;
  }

  getPaddingText() {
    return `padding: ${this.padding}`;
  }

  getStrideText() {
    return `stride: ${this.stride}`;
  }

  getChannelText() {
    return `channel: ${this.channel}`;
  }
}

const NeuralNet = () => {
  const [data, setData] = useState([new Layer(0)]);
  const svgRef = useRef(null);

  useEffect(() => {
    const svg = select(svgRef.current); // selection 객체

    svg.selectChildren().remove();

    svg
      .selectAll(".neural-net")
      .data(data)
      .join(
        enter => enter.append("rect"),
        update => update.attr("class", "updated"),
        exit => exit.remove()
      )
      .attr("x", layer => layer.index * 50)
      .attr("y", 100)
      .attr("width", 30)
      .attr("height", 70)
      .attr("stroke", "black")
      .attr("fill", "#69a3b2")
      .on("hover", () => {
        console.log("hover");
      });

    svg
      .selectAll(".neural-net")
      .data(data)
      .join(
        enter => enter.append("text"),
        update => update.attr("class", "updated"),
        exit => exit.remove()
      )
      .attr("dy", "0em") // you can vary how far apart it shows up
      .text(layer => layer.getPaddingText())
      .attr("x", layer => layer.index * 50 + 25)
      .attr("y", 135)
      .attr("font-family", "sans-serif")
      .attr("font-size", "10px")
      .attr("fill", "black")
      .attr("text-anchor", "middle");

    svg
      .selectAll(".neural-net")
      .data(data)
      .join(
        enter => enter.append("text"),
        update => update.attr("class", "updated"),
        exit => exit.remove()
      )
      .attr("dy", "1em") // you can vary how far apart it shows up
      .text(layer => layer.getStrideText())
      .attr("x", layer => layer.index * 50 + 25)
      .attr("y", 135)
      .attr("font-family", "sans-serif")
      .attr("font-size", "10px")
      .attr("fill", "black")
      .attr("text-anchor", "middle");

    svg
      .selectAll(".neural-net")
      .data(data)
      .join(
        enter => enter.append("text"),
        update => update.attr("class", "updated"),
        exit => exit.remove()
      )
      .attr("dy", "2em") // you can vary how far apart it shows up
      .text(layer => layer.getChannelText())
      .attr("x", layer => layer.index * 50 + 25)
      .attr("y", 135)
      .attr("font-family", "sans-serif")
      .attr("font-size", "10px")
      .attr("fill", "black")
      .attr("text-anchor", "middle");
  }, [data]);

  return (
    <>
      <svg ref={svgRef} className="neural-net"></svg>
      <button
        onClick={() => {
          const newData = data.concat([new Layer(data.length)]);
          setData(newData);
        }}
      >
        add layer
      </button>
      <button
        onClick={() => {
          setData(data.filter(layer => layer.index < 3));
        }}
      >
        delete layers after 3
      </button>
    </>
  );
};

export default NeuralNet;
